#!/usr/bin/env bash

# Exit on any error
tset -e

# Configuration variables
DATA_ROOT="./data"
MODEL_DIRS=(without_watermark revmark rivagan videoseal videoshield videomark)
SCORES_FILE="./results/gpt_4o_video_watermark_scores_modelscope.json"
EVAL_FILE="./results/eval_gpt_4o_video_watermark_scores_modelscope.json"
TOKENS=(sk-yourtoken1 sk-yourtoken2 sk-yourtoken3)

python .scripts/eval_video_quality/eval_video_quality_by_mllm.py \
  --data-root "${DATA_ROOT}" \
  --model-dirs "${MODEL_DIRS[@]}" \
  --scores-file "${SCORES_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --tokens "${TOKENS[@]}"

