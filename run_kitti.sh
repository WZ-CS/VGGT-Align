#!/usr/bin/env bash

IMAGE_DIR="./"
EXP_ROOT="./exps"

CUDA_VISIBLE_DEVICES=0 python vggt_align.py \
    --image_dir "${IMAGE_DIR}" \
    --exp_dir "${EXP_ROOT}" \
    --ground_prior \
    --camera_height 1.65 \
    --tta \
    --tta_steps 3