#!/usr/bin/env python

# ----------------------------------------------------------------------
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------

import autograd.numpy as np
import sys

from autograd import grad, jacobian

np.set_printoptions(linewidth=1000)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: format(x, '11.4e')})

# Applies softmax per every column of the 'x' matrix.
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)

# Multi-head attention model, the forward pass.  Works with multiple 
# 'q' vector time-steps and one sequence (batchSize=1/beamSize=1).
def attn_fwd(q, k, v, wq, wk, wv, wo, nheads, rlink):
    out = 0

    # Threshold to detect a single spike softmax output.  When 
    # the softmax output is close to 1.0 at one location and 
    # zeros or very small values elsewhere, in the backward pass 
    # the Jacobian of the softmax block is all zeros or very 
    # small values - basically noise.  In such a case the multi-
    # head attention dgrad/wgrad results will be inaccurate.
    threshold = 1.0 / np.shape(k)[1]

    for i in range(nheads):
        q_bar = np.dot(wq[i,:,:], q)
        k_bar = np.dot(wk[i,:,:], k)
        v_bar = np.dot(wv[i,:,:], v)

        beta = scaler * np.dot(k_bar.transpose(), q_bar)
        alpha = softmax(beta)

        chk = np.sum(alpha > threshold, axis=0)
        chk = (chk == 1)
        if np.all(chk):
            print("WARNING: dgrad/wgrad may be inaccurate")

        h_bar = np.dot(v_bar, alpha)
        h = np.dot(wo[i,:,:], h_bar)

        out = np.add(out, h)

    # Residual connection from 'q' to 'out'.
    if rlink != 0:
        out = np.add(out, q)

    return out

# Embeddings are saved in column vectors.
def load_data(tag, fname, w, dim = 2):
    d = np.loadtxt(fname, ndmin=2)
    if (np.shape(d)[0] != np.shape(w)[dim]):
        print("Number of rows=%d in file '%s' does not match weight cols=%d" % (np.shape(d)[0], fname, np.shape(w)[dim]))
        sys.exit()
    print("Loaded [%s] of %s data from '%s'" % ("x".join(map(str, np.shape(d))), tag, fname))
    return d

# Split weights vertically into 'nheads' matrices.
def load_weights(tag, fname, nheads):
    w = np.loadtxt(fname, ndmin=2) 
    if (np.shape(w)[0] % nheads != 0):
        print("Number of rows=%d is not divisible by nheads=%d in file '%s'" % (np.shape(w)[0], nheads, fname))
        sys.exit()
    w = np.reshape(w, (nheads, np.shape(w)[0] / nheads, np.shape(w)[1]))
    print("Loaded [%s] of %s weights from '%s'" % ("x".join(map(str, np.shape(w))), tag, fname))
    return w

# Compares 'a' result with 'b' reference
def compare(tag, a, b, rtol, atol):
    a = a.flatten()
    b = b.flatten()

    # Same condition as allclose() but with indices.
    tol = atol + rtol * abs(b)
    indices = np.transpose(np.where(abs(a - b) > tol))
    status = "PASS" if len(indices) == 0 else "FAIL"

    rel_err = max(np.abs(a - b) / b)
    abs_err = max(np.abs(a - b))
    print("%s: %s [rel_err=%.6e abs_err=%.6e, rtol=%.6e, atol=%.6e]" % (status, tag, rel_err, abs_err, rtol, atol))
    for i in indices:
        print("%s_res[%d]=%+.6e, %s_ref[%d]=%+.6e" % (tag, i, a[i], tag, i, b[i]))
    print("")

# ----
# Main
# ----

bar_len=70

train, nheads, scaler, rlink = np.loadtxt('meta.dat', unpack=True)
train  = int(train)
nheads = int(nheads)
rlink  = int(rlink)

print("%s" % ('=' * bar_len))
print("Multi-head attention model: train=%d, nheads=%d, scaler=%.4e" % (train, nheads, scaler))
print("%s\n" % ('=' * bar_len))

wq = load_weights('WQ', 'wq.dat', nheads)
wk = load_weights('WK', 'wk.dat', nheads)
wv = load_weights('WV', 'wv.dat', nheads)
wo = load_weights('WO', 'wo.dat', nheads)
q  = load_data('Q', 'q.dat', wq)
k  = load_data('K', 'k.dat', wk)
v  = load_data('V', 'v.dat', wv)

if (train != 0):
    dout = load_data('DOUT', 'dout.dat', wo, 1)

# Compute the multi-head attention forward response.
out = attn_fwd(q, k, v, wq, wk, wv, wo, nheads, rlink)

# Generate functions to compute Jacobians vs each trainable input.
# One argument per jacobian() function due to a bug in numpy/autograd.
# See: Top-level Jacobian only works with integer argnum
# https://github.com/XanaduAI/pennylane/issues/112

if (train != 0):
    attn_jaco_q  = jacobian(attn_fwd, argnum=0)
    attn_jaco_k  = jacobian(attn_fwd, argnum=1)
    attn_jaco_v  = jacobian(attn_fwd, argnum=2)
    attn_jaco_wq = jacobian(attn_fwd, argnum=3)
    attn_jaco_wk = jacobian(attn_fwd, argnum=4)
    attn_jaco_wv = jacobian(attn_fwd, argnum=5)
    attn_jaco_wo = jacobian(attn_fwd, argnum=6)

    dq = attn_jaco_q(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dq = np.tensordot(dout, dq, axes=([0,1],[0,1]))

    dk = attn_jaco_k(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dk = np.tensordot(dout, dk, axes=([0,1],[0,1]))

    dv = attn_jaco_v(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dv = np.tensordot(dout, dv, axes=([0,1],[0,1]))

    dwq = attn_jaco_wq(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dwq = np.tensordot(dout, dwq, axes=([0,1],[0,1]))

    dwk = attn_jaco_wk(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dwk = np.tensordot(dout, dwk, axes=([0,1],[0,1]))

    dwv = attn_jaco_wv(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dwv = np.tensordot(dout, dwv, axes=([0,1],[0,1]))

    dwo = attn_jaco_wo(q, k, v, wq, wk, wv, wo, nheads, rlink)
    dwo = np.tensordot(dout, dwo, axes=([0,1],[0,1]))

print("\n%s" % ('=' * bar_len))
print("Checking results from cuDNN library")
print("%s\n" % ('=' * bar_len))

out_lib = load_data('OUT', 'out.dat', wo, 1)
compare('out', out_lib, out, 1e-4, 1e-3)

if (train != 0):
    dq_lib = load_data('DQ', 'dq.dat', wq)
    # In cuDNN, the residual input can be a different input from 'q'. 
    # In the cuDNN dgrad call, 'dout' is not copied to 'dres'.  When 
    # the residual connection is enabled and 'res=q', we need to add 
    # 'dout' to 'dq' to account for two paths from 'q' to 'out'.
    if rlink != 0:
        dq_lib = np.add(dq_lib, dout)
    compare('dq', dq_lib, dq, 1e-4, 1e-3)

    dk_lib = load_data('DK', 'dk.dat', wk)
    compare('dk', dk_lib, dk, 1e-4, 1e-3)

    dv_lib = load_data('DV', 'dv.dat', wv)
    compare('dv', dv_lib, dv, 1e-4, 1e-3)

    dwq_lib = load_weights('DWQ', 'dwq.dat', nheads)
    compare('dwq', dwq_lib, dwq, 1e-4, 1e-3)

    dwk_lib = load_weights('DWK', 'dwk.dat', nheads)
    compare('dwk', dwk_lib, dwk, 1e-4, 1e-3)

    dwv_lib = load_weights('DWV', 'dwv.dat', nheads)
    compare('dwv', dwv_lib, dwv, 1e-4, 1e-3)

    dwo_lib = load_weights('DWO', 'dwo.dat', nheads)
    compare('dwo', dwo_lib, dwo, 1e-4, 1e-3)

