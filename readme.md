# 🎬 VideoMark

**VideoMark**: *A Distortion-Free Robust Watermarking Framework for Video Diffusion Models*  
[Official Implementation] of the paper.

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

Make sure you have Python 3.10+ installed. Then run:
```shell

pip install -r requirements.txt
```

### 2️⃣ Generate a Watermark Key
```shell
python PRC_key_gen.py --hight 512 --width 512 --fpr 0.01 --prc_t 3
```

### 3️⃣ Watermark Embedding and Extraction
```shell
python embedding_and_extraction.py \
        --model_name i2vgen-xl \
        --num_frames 16 \
        --num_bit 512 \
        --num_inference_steps 50 \
        --output_dir <your save dir> \
        --keys_path  <your keys path>\
```

### 4️⃣ Robustness Test
```shell
python temporal_tamper.py
        --model_name i2vgen-xl \
        --num_bit 512 \
        --num_inference_steps 50 \
        --video_frames_dir <your dir> \
        --keys_path  <your keys path> \
```
---

## 📊 Video Quality Evaluation

To evaluate the quality of watermarked videos, you can perform both **objective** and **subjective** assessments.

### 🧪 Objective Evaluation with VBench

We recommend using [VBench](https://github.com/Vchitect/VBench) — VBench: Comprehensive Benchmark Suite for Video Generative Models


### 👁️ Subjective Evaluation

For subjective assessments, we provide sample videos and guidelines in the following folder:

```shell
cd eval_quality
```