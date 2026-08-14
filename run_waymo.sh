#!/usr/bin/env bash

IMAGE_DIR="./data/waymo/segment/images"
EXP_ROOT="./exps"

CUDA_VISIBLE_DEVICES=0 python vggt_align.py \
  --image_dir "${IMAGE_DIR}" \
  --config "./configs/waymo.yaml" \
  --exp_dir "${EXP_ROOT}" \
  --ground_prior \
  --camera_height 2.05 \
  --road_width_prior \
  --adaptive_blend \
  --tta \
  --tta_steps 3
