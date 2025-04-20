JSON_PATH=".results/gpt_4o_video_watermark_scores_i2vgen-xl_v2.json"

# Execute the Python average score calculator
python ./scripts/eval_video_quality/scores_for_statistical_assessments.py --json-path "${JSON_PATH}"