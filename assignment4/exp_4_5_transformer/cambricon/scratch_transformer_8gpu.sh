#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../;pwd)

pushd $TRANS_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --distributed
popd
