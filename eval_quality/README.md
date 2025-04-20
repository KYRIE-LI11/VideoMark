## 📌Evaluation

### Video Quality Evaluation Using MLLM

- We provide the example that score video quality by GPT-4o

- Reference to the relevant file directory structure at the time of evaluation
```shell
├── data
│   ├── videoshield
        ├── modelscope
        ├── i2vgen-xl
            ├── a_ballerina_practicing_in_the_dance_studio_epoch0
            ...
            ├── a_ballerina_practicing_in_the_dance_studio_epoch9
                ├── wm.mp4
...
│   ├── videomark
        ├── modelscope
        ├── i2vgen-xl
            ├── a_ballerina_practicing_in_the_dance_studio_epoch0
            ...
            ├── a_ballerina_practicing_in_the_dance_studio_epoch9
                ├── wm.mp4
...

├── results
│   ├── gpt_4o_video_watermark_scores_i2vgen-xl_v2.json
│   └── gpt_4o_video_watermark_scores_modelscope_v2.json
└── scripts
    └── eval_video_quality
        ├── count_best_methods.py
        ├── eval_i2v_video_quality_by_gpt-4o.sh
        ├── eval_modelscope_video_quality_by_gpt-4o.sh
        ├── eval_video_quality_by_mllm.py
        ├── run_count_best_methods_i2v.sh
        ├── run_count_best_methods_modelscope.sh
        ├── scores_for_statistical_assessments_i2v.sh
        ├── scores_for_statistical_assessments_modelscope.sh
        └── scores_for_statistical_assessments.py
```

- You need to change the API website in the line 159 of ```./scripts/eval_video_quality/eval_video_quality_by_mllm.py```, and you need to input your token in the file ```./scripts/eval_video_quality/eval_modelscope_video_quality_by_gpt-4o.sh```.
then run the command:
```shell
# 1. Use GPT-4o to score video quality across multiple dimensions
bash ./scripts/eval_video_quality/eval_modelscope_video_quality_by_gpt-4o.sh

# 2. Statistical GPT-4o's preference for different methods for the same sample
bash ./scripts/eval_video_quality/scores_for_statistical_assessments_modelscope.sh

# 3. Calculate the average of the test samples across dimensions
bash ./scripts/eval_video_quality/scores_for_statistical_assessments_modelscope.sh
```