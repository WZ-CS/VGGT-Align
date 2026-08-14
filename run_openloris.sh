#!/usr/bin/env bash

IMAGE_DIR="./data/openloris/office"
EXP_ROOT="./exps"

CUDA_VISIBLE_DEVICES=0 python vggt_align.py \
  --image_dir "${IMAGE_DIR}" \
  --config "./configs/openloris.yaml" \
  --exp_dir "${EXP_ROOT}" \
  --ground_prior \
  --camera_height 1.65 \
  --road_width_prior \
  --tta \
  --tta_steps 3
