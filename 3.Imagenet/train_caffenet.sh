#!/usr/bin/env sh
set -e

../caffe/build/tools/caffe train \
    --solver=./bvlc_reference_caffenet/solver.prototxt $@
