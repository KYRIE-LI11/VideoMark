JSON_PATH="./results/gpt_4o_video_watermark_scores_i2vgen-xl_v2.json"
METHODS=(revmark rivagan videoseal videoshield videomark)

python ./scripts/eval_video_quality/count_best_methods.py --json-path "$JSON_PATH" --methods "${METHODS[@]}"