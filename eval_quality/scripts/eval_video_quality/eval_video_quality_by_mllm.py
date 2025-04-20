import os
import cv2
import numpy as np
import base64
import json
import re
import logging
from tqdm import tqdm
from glob import glob
from openai import OpenAI
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video watermark removal results using GPT-4o")
    parser.add_argument('--data-root', default='./data', help='Root directory containing model output data')
    parser.add_argument('--model-dirs', nargs='+', default=['revmark', 'rivagan', 'videoseal', 'videoshield', 'videomark'], help='List of model directory names under data root')
    parser.add_argument('--scores-file', default='./results/gpt_4o_video_watermark_scores_modelscope.json', help='Path to save intermediate scores JSON file')
    parser.add_argument('--eval-file', default='./results/eval_gpt_4o_video_watermark_scores_modelscope.json', help='Path to save final evaluation JSON file')
    parser.add_argument('--tokens', nargs='+', required=True, help='List of OpenAI API tokens to use for requests')
    return parser.parse_args()

args = parse_args()

DATA_ROOT = args.data_root
MODEL_DIRS = args.model_dirs
SCORES_FILE = args.scores_file
EVAL_FILE = args.eval_file
TOKEN_LIST = args.tokens

# Global lock to ensure thread-safe writes to scores
lock = threading.Lock()

def sample_frames(video_path: str, start_time: float = 0, end_time: Optional[float] = None, num_frames: int = 16) -> list:
    """Extract key frames from the video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames
    end_frame = min(end_frame, total_frames)
    frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
    frames = []
    with tqdm(frame_indices, desc="🔄 Extracting key frames", unit="frame", leave=False) as pbar:
        for idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
            pbar.set_postfix({"Current frame": f"{idx}/{end_frame}"})
    cap.release()
    return frames

def encode_frames(frames: List[Any]) -> List[str]:
    """
    Convert sampled frames to a list of base64-encoded JPEG strings
    """
    encoded_frames = []
    for frame in frames:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(jpg_as_text)
    return encoded_frames

def construct_prompt(model_name: str, sample_name: str) -> str:
    """
    Construct the scoring prompt, requesting GPT-4o to return a JSON object with scores:
    {
        "subject_consistency": <0-10>,
        "background_consistency": <0-10>,
        "motion_smoothness": <0-10>,
        "imaging_quality": <0-10>,
        "total_score": <number>
    }
    """
    prompt = (
        f"Please perform a thorough and objective evaluation of the video result for sample '{sample_name}'. "
        "Using the provided 16 sampled frames as reference, carefully assess the video on the following criteria:\n\n"
        "1. Subject Consistency: Evaluate how consistently and accurately the subject is represented across all frames.\n"
        "2. Background Consistency: Assess the uniformity and coherence of the background elements throughout the video.\n"
        "3. Motion Smoothness: Judge the fluidity and natural progression of motion, ensuring transitions are smooth and realistic.\n"
        "4. Imaging Quality: Evaluate the overall visual quality, including clarity, sharpness, color fidelity, and detail.\n\n"
        "For each criterion, assign a numeric score between 0 (lowest) and 10 (highest) based solely on the visual data. "
        "Compute the total score as the sum of these individual scores. "
        "Your evaluation should be objective, data-driven, and free from any additional commentary or explanation. "
        "Return your response strictly as a JSON object formatted exactly as shown below without any extra text:\n\n"
        '{\n'
        '    "subject_consistency": <number>,\n'
        '    "background_consistency": <number>,\n'
        '    "motion_smoothness": <number>,\n'
        '    "imaging_quality": <number>,\n'
        '    "total_score": <number>\n'
        '}\n'
    )
    return prompt

def parse_score_with_regex(text: str) -> Optional[Dict]:
    """
    Use regex to extract scores from the returned text, supporting float numbers.
    """
    keys = ["subject_consistency", "background_consistency", "motion_smoothness", "imaging_quality", "total_score"]
    scores = {}
    for key in keys:
        pattern = fr'"{key}"\s*:\s*([\d\.]+)'
        match = re.search(pattern, text)
        if match:
            try:
                scores[key] = float(match.group(1))
            except ValueError:
                logging.error(f"Failed to convert {key} score.")
                scores[key] = None
        else:
            logging.error(f"Could not capture {key} score.")
            scores[key] = None
    if any(v is not None for v in scores.values()):
        return scores
    return None

def score_video(video_path: str, model_name: str, sample_name: str) -> Optional[Dict]:
    """
    Sample video frames, construct the prompt, call the API to score, and try fallback tokens or regex parsing if needed.
    """
    logging.info(f"Scoring video: {video_path}")
    frames = sample_frames(video_path, num_frames=16)
    if not frames:
        logging.error(f"Cannot read video frames: {video_path}")
        return None
    encoded_frames = frames
    prompt = construct_prompt(model_name, sample_name)
    headers = {"Content-Type": "application/json"}
    messages = [
        {
            "role": "system",
            "content": (
                "You are now a professional video quality assessment expert, familiar with various metrics including "
                "video content, image quality, motion smoothness, and background consistency. Based on the video keyframes "
                "and related descriptions provided by the user, please provide a detailed score for the video."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in encoded_frames]
            ]
        }
    ]
    last_error = None
    for token in TOKEN_LIST:
        client = OpenAI(base_url="https://aizex.top/v1", api_key=token)
        headers["Authorization"] = f"Bearer {token}"
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=128)
            result = response.choices[0]
            content = result.message.content[7:-4]  # strip ```json and ```
            try:
                score_data = json.loads(content)
                return score_data
            except Exception as parse_error:
                logging.error(f"Direct JSON parsing failed: {parse_error}. Response content: {content}")
                regex_result = parse_score_with_regex(content)
                if regex_result:
                    logging.info("Successfully captured scores via regex.")
                    return regex_result
                else:
                    logging.error("Regex parsing also failed.")
                    last_error = parse_error
                    continue
        except Exception as e:
            if "timeout" in str(e).lower() or "bad gateway" in str(e).lower():
                logging.error(f"Token {token} request timed out or Bad Gateway error, trying next token.")
            else:
                logging.error(f"Exception using token {token} to score video: {str(e)}")
            last_error = e
            continue
    logging.error(f"All tokens failed, last error: {last_error}")
    return None

def load_json_file(file_path: str) -> Dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
    return {}

def save_json_file(data: Dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def save_scores(scores: Dict) -> None:
    save_json_file(scores, SCORES_FILE)

def save_evaluation(eval_results: Dict) -> None:
    save_json_file(eval_results, EVAL_FILE)

def process_video(sample_name: str, model: str, video_path: str) -> Optional[Dict]:
    """
    Single video scoring task, returns the scoring result
    """
    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return None
    result = score_video(video_path, model, sample_name)
    return result

def main():
    sample_sets: Dict[str, Dict[str, str]] = {}
    for model in MODEL_DIRS:
        model_path = os.path.join(DATA_ROOT, model)
        sample_dirs = [d for d in glob(os.path.join(model_path, "modelscope/**/*epoch9"), recursive=True) if os.path.isdir(d)]
        for d in sample_dirs:
            sample_name = os.path.basename(d)
            if sample_name.endswith("epoch9"):
                sample_name = sample_name[:-len("_epoch9")]
            sample_sets.setdefault(sample_name, {})[model] = d
    valid_samples = {k: v for k, v in sample_sets.items() if len(v) == len(MODEL_DIRS)}
    logging.info(f"Found {len(valid_samples)} valid samples")
    scores = load_json_file(SCORES_FILE)
    futures_mapping = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for sample_name, model_dirs in valid_samples.items():
            with lock:
                if sample_name not in scores:
                    scores[sample_name] = {}
            for model, dir_path in model_dirs.items():
                video_path = os.path.join(dir_path, "wm.mp4")
                if not os.path.exists(video_path):
                    logging.error(f"Video not found: {video_path}")
                    with lock:
                        scores[sample_name][model] = None
                    continue
                with lock:
                    if model in scores[sample_name] and scores[sample_name][model] is not None:
                        continue
                future = executor.submit(process_video, sample_name, model, video_path)
                futures_mapping[future] = (sample_name, model)
        for future in as_completed(futures_mapping, timeout=None):
            sample_name, model = futures_mapping[future]
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Task exception: {e}")
                result = None
            with lock:
                if sample_name not in scores:
                    scores[sample_name] = {}
                scores[sample_name][model] = result
                save_json_file(scores, SCORES_FILE)
                logging.info(f"Scoring result for sample {sample_name}, model {model} has been saved.")
    eval_results = {}
    for sample_name, model_scores in scores.items():
        best_model = None
        best_score = -float("inf")
        for model, score_result in model_scores.items():
            if score_result is None:
                continue
            total = score_result.get("total_score")
            try:
                total = float(total)
            except Exception as e:
                logging.error(f"Failed to convert total_score: {e}")
                total = 0
            if total > best_score:
                best_score = total
                best_model = model
        eval_results[sample_name] = {"best_model": best_model, "best_score": best_score}
    save_json_file(eval_results, EVAL_FILE)
    logging.info("Final evaluation results saved to evaluation file")

if __name__ == "__main__":
    main()
