# CUDA Support for Bazel

This repo is deprecated in favor of rules_cuda. Ref: https://github.com/tensorflow/runtime/tree/master/third_party/rules_cuda

## Introduction

As of Bazel 3.5.0, there is still no native CUDA support in Bazel (Refer to
[Bazel issue: Native CUDA Support](https://github.com/bazelbuild/bazel/issues/6578)
to see why). So I borrowed CUDA support code for Bazel from the
[TensorFlow](https://github.com/tensorflow/tensorflow.git) project.

The mechanism for this support was Bazel's custom toolchain support. Refer to
[tutorial-cc-toolchain-config](https://docs.bazel.build/versions/master/tutorial/cc-toolchain-config.html)
for an thorough understanding on Bazel's `cc-toolchain-config`.

## Warnings Ahead

Note that CUDA support for Bazel IN THIS REPO was tailored to run on Linux ONLY.
Windows and MacOS support are stripped off which are present in the original
TensorFlow repo.

As far as CPU architecture is concerned, ONLY `x86_64`, `arm` and `aarch64`
support are reserved for the editor's own need. If you need support for a
different Arch, please refer to
[TensorFlow](https://github.com/tensorflow/tensorflow.git).

The author also adjusted default CUDA compute capability settings. Only newer
NVidia GPU Cards with SM values = [6.0, 6.1, 7.0, 7.2, 7.5, ...] are supported,
and requires CUDA Toolkit with version >= 9.0.

## Status

**CUDA & cuDNN & TensorRT examples works now**

As of Fri June 16 18:12:47 CST 2020, this project is still a Work-In-Progress,
with the following TODO list:

- NCCL support was not tailored yet.
- BUILD file for cudnn-examples/mnistCUDNN

### Stripped Modules/Dependencies/Options

- [gemmlowp](https://github.com/google/gemmlowp)
- [mkl](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
- OpenCL
- [ROCm](https://github.com/RadeonOpenCompute/ROCm)
- [XLA](https://www.tensorflow.org/xla)
- [Intel nGraph support](https://github.com/NervanaSystems/ngraph)
- Option for experimental clang download

### References

- TF: third_party/mlir/tblgen.bzl
- TF: tensorflow/workspace.bzl
- TF: tensorflow/tensorflow.bzl
- [Deprecated] https://github.com/Hibbert-pku/bazel_nvcc.git

## How to Run

#### Step 1: Run `./bootstrap.sh --noninteractive` or `./bootstrap.sh` to configure your bazel build options.

#### Step 2: Install Bazel 3.2+.

Ref: [GitHub Bazel Release Page](https://github.com/bazelbuild/bazel/releases)

All programs were tested on an Ubuntu-18.04 x86_64 machine with one NVidia GTX
1070 GPU card, using Bazel 3.2.0.

#### Step 3: Run the `hello-world` example to check if everything works fine.

```
bazel build --config=opt //...
bazel run //src:hello_world
```

## Welcome On Board with Bazel!
