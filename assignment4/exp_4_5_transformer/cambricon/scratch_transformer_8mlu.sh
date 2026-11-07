#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../;pwd)

pushd $TRANS_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --distributed --cnmix --opt_level O0
popd
