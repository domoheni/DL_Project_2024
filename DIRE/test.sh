#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire


EXP_NAME="stanford_cars"
DATASETS="stanford_cars"
DATASETS_TEST="stanford_cars"

python test.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST

