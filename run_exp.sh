#!/bin/bash

BS=64
GPU_ID=0
MAX_ITER=1000
DATA_PATH=./data

if [ $1 == 'mnist' ]; then
    DATASET=mnist
    DATAROOT=${DATA_PATH}/mnist
    ISIZE=32
    NC=1
    NZ=10
elif [ $1 == 'cifar10' ]; then
    DATASET=cifar10
    DATAROOT=${DATA_PATH}/cifar10
    ISIZE=32
    NC=3
    NZ=128
elif [ $1 == 'celeba' ]; then
    DATASET=celeba
    DATAROOT=${DATA_PATH}/celeba/splits
    ISIZE=64
    NC=3
    NZ=64
elif [ $1 == 'lsun' ]; then
    DATASET=lsun
    DATAROOT=${DATA_PATH}/lsun
    ISIZE=64
    NC=3
    NZ=128
else
    echo "unknown dataset [mnist /cifar10 / celeba / lsun]"
    exit
fi

EXP_FILE="${DATASET}_mmd-gan"
LOG_FILE="${DATASET}_mmd-gan.log"

cmd="stdbuf -o L python mmd_gan.py --dataset ${DATASET} --dataroot ${DATAROOT} --batch_size ${BS} --image_size ${ISIZE} --nc ${NC}  --nz ${NZ} --max_iter ${MAX_ITER} --gpu_device ${GPU_ID} --experiment ${EXP_FILE} | tee ${LOG_FILE}"

echo $cmd
eval $cmd
