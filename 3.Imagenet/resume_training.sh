#!/usr/bin/env sh
set -e

../caffe/build/tools/caffe train \
    --solver=./bvlc_reference_caffenet/solver.prototxt \
    --snapshot=./bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5 \
    $@
