
<div align="center">
<h2><a href="https://arxiv.org/abs/2504.16359" style="color:#68edcb">VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models</a></h2>
        If our project helps you, please give us a star ⭐ on GitHub to support us. 🧙🧙
        
[![arXiv](https://img.shields.io/badge/arXiv-2504.16359-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2504.16359) 
</div>


## 🔥 News
* **`2025.0x.xx`** 🎉 We are happy to announce that we release our code.
* **`2025.04.23`** 🌟 We released the paper [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359).
<div align="center"><img src="https://github.com/KYRIE-LI11/VideoMark/blob/main/docs/overall_pipeline.png" width="800" /><div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;">The overall framework of VideoMark. </div></div>


----


## 🛠️ Requirements and Installation

### Install Dependencies
Basic Dependencies:
* Python >= 3.10
* ···
Then run:
```shell
pip install -r requirements.txt
```

### Generate a Watermark Key
```shell
python PRC_key_gen.py --hight 512 --width 512 --fpr 0.01 --prc_t 3
```

### Watermark Embedding and Extraction
```shell
python embedding_and_extraction.py \
        --model_name i2vgen-xl \
        --num_frames 16 \
        --num_bit 512 \
        --num_inference_steps 50 \
        --output_dir <your save dir> \
        --keys_path  <your keys path>\
```

### Robustness Test
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

For subjective assessments, we provide sample videos and guidelines in the following [`folder: eval_quality`](https://github.com/KYRIE-LI11/VideoMark/blob/main/eval_quality/README.md):
```shell
cd eval_quality
```


## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=KYRIE-LI11/VideoMark&type=Date)](https://star-history.com/#KYRIE-LI11/VideoMark&Date)


## 📑 Citation
If you find VideoMark useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{hu2025videomark,
  title={VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models},
  author={Hu, Xuming and Li, Hanqian and Li, Jungang and Liu, Aiwei},
  journal={arXiv preprint arXiv:2504.16359},
  year={2025}
}
```
