#!/usr/bin/env bash

# Stop after the first error.
set -e

# Echo commands that are executed.
set -x

API_TST=./multiHeadAttention

if [ ! -x $API_TST ]; then
    echo "Executable file '$API_TST' not found in the current folder."
    exit 1
fi

if [ -z "$1" ]; then
SEED=1234
else
SEED=$1
fi

rm -f meta.dat q.dat k.dat v.dat out.dat wq.dat wk.dat wv.dat wo.dat
rm -f dout.dat dq.dat dk.dat dv.dat dwq.dat dwk.dat dwv.dat dwo.dat

$API_TST -attnFileDump1 -attnTrain1 -attnDataType0 -attnResLink1 -attnDataLayout3 -attnNumHeads3 -attnBeamSize1 -attnBatchSize1 -attnQsize8 -attnKsize8 -attnVsize8 -attnProjQsize2 -attnProjKsize2 -attnProjVsize2 -attnProjOsize8 -attnResLink0 -attnSeqLenQ4 -attnSeqLenK10 -attnSmScaler1.0 -attnRandSeed$SEED

./attn_ref.py

