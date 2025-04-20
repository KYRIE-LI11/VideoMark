import os
import argparse
import copy
import cv2
import torch
import numpy as np
import random
from utils import transform_video, cv2_to_pil, get_video_latents, recover
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str, Detect, Decode
import src.pseudogaussians as prc_gaussians
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline,I2VGenXLPipeline, DDIMInverseScheduler
import multiprocessing
from multiprocessing import Pool, cpu_count
from PIL import Image
import pickle
from Levenshtein import distance
import json
from tqdm import tqdm
from Levenshtein import distance as lev_distance
np.random.seed(520)

def bits_to_string(bits):
    return ''.join(map(str, bits))

def get_label_index(i, length=16):
    return [(i + j) for j in range(length)]


def process_frame(frame_index, reversed_latents_w_cpu, decoding_key, message_bits):
    reversed_prc = prc_gaussians.recover_posteriors(
        reversed_latents_w_cpu[:,:,frame_index].to(torch.float64).flatten(), 
        variances=1.5
    ).flatten()
    
    detection_result = Detect(decoding_key, reversed_prc, false_positive_rate=0.01)
    message_placeholder = '<message_placeholder>'
    if not detection_result:
        decode_message = np.full((1,len(message_bits[0])), -1)
        decode_message_str = message_placeholder
    else:
        #detection_value = 1
        decode_message = Decode(decoding_key, reversed_prc)
        decode_message_str = bits_to_string(decode_message)
    distances = np.array([distance(decode_message_str, msg) for msg in message_bits])
    min_distance = np.min(distances)
    if len(np.unique(distances)) == 1:
        idx = -1
    else:
        idx = np.argmin(distances)
    return decode_message, min_distance, idx

def process_frames_in_parallel(num_frames, reversed_latents_w, decoding_key, message_bits):
    reversed_latents_w_cpu = reversed_latents_w.cpu()
    num_bit = message_bits.shape[1]
    template_bit  = [bits_to_string(msg) for msg in message_bits]
    message_list = []
    idx_list = []
    distance_list = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_frame, 
                               [(frame_index, reversed_latents_w_cpu, decoding_key, template_bit) 
                                for frame_index in range(1,num_frames)])
    
    for decode_message, distance, idx in results:
        message_list.append(decode_message)
        distance_list.append(distance)
        idx_list.append(idx)

    # Sort the indices and keep the order of the original list
    recovered_index, recovered_message, recovered_distance = recover(idx_list, message_list, distance_list,num_bit)
    return recovered_index, recovered_message, recovered_distance

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
    
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def temporal_tamper(video_frames, tampering_type_list, shift_value, message_bits_sequence):
    tampered_videos = {}
    video_frames_ids = get_label_index(shift_value, length=len(video_frames))
    message_bits_label = message_bits_sequence[shift_value:shift_value+len(video_frames)].tolist()
    for tampering_type in tampering_type_list:
        if tampering_type == 'identity':
            tampered_videos['identity'] = [copy.deepcopy(video_frames), copy.deepcopy(video_frames_ids)]
        if tampering_type == 'frame swap':
            frame_swap_steps = 4
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            message_bits_tampered = copy.deepcopy(message_bits_label)

            for i in range(1, len(video_frames) - 1, frame_swap_steps):
                video_frames_tampered[i] = video_frames[i + 1]
                video_frames_tampered[i + 1] = video_frames[i]
            tampered_videos['frame swap'] = [video_frames_tampered, video_frames_tampered_ids, np.array(message_bits_tampered)]
        if tampering_type == 'frame insert':
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            message_bits_tampered = copy.deepcopy(message_bits_label)
            insert_id = random.randint(1, len(video_frames))
            insert_id_frame_list = [
                [video_frames_ids[insert_id-1], video_frames[insert_id-1], message_bits_tampered[insert_id-1]],
            ]
            insert_id_frame = random.choice(insert_id_frame_list)
            video_frames_tampered_ids.insert(insert_id, insert_id_frame[0])
            video_frames_tampered.insert(insert_id, insert_id_frame[1])

            message_bits_tampered.insert(insert_id, insert_id_frame[2])
            tampered_videos['frame insert'] = [video_frames_tampered, video_frames_tampered_ids, np.array(message_bits_tampered)]
        if tampering_type == 'frame drop':
            video_frames_tampered = copy.deepcopy(video_frames)
            video_frames_tampered_ids = copy.deepcopy(video_frames_ids)
            message_bits_tampered = copy.deepcopy(message_bits_label)

            drop_id = random.randint(1, len(video_frames)-1)
            del video_frames_tampered[drop_id]
            del video_frames_tampered_ids[drop_id]
            del message_bits_tampered[drop_id]
            tampered_videos['frame drop'] = [video_frames_tampered, video_frames_tampered_ids,np.array(message_bits_tampered)]
    return tampered_videos

def simulate_one_round(args):
    null_messages, template_bit, test_distance = args
    null_strings = [bits_to_string(null_msg) for null_msg in null_messages]
    
    min_distances = []
    for null_str in null_strings:
        distances = [lev_distance(null_str, tmpl) for tmpl in template_bit]
        min_distances.append(min(distances))
    
    avg_min_distance = np.mean(min_distances)
    #print(avg_min_distance)
    return 0 if test_distance <= avg_min_distance else 1

def main(args):
    model_name = args.model_name
    t = args.threshold
    num_bit = args.num_bit
    message_bits_sequence = np.random.randint(0, 2, size=(500,num_bit))
    T = args.resample_num
    if args.model_path is not None:
        model_path = args.model_path
    else:
        if model_name == 'modelscope':
            model_path = 'damo-vilab/text-to-video-ms-1.7b'
        elif model_name == 'i2vgen-xl':
            model_path = 'ali-vilab/i2vgen-xl'
        else:
            raise ValueError

    num_inversion_steps = args.num_inversion_steps
    device = args.device
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
    )
    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)

        height = 64
        width = 64
    elif model_name == 'i2vgen-xl':
        video_pipe = I2VGenXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        height = 64
        width = 64
    else:
        raise ValueError
    
    keys_path = os.path.join(args.keys_path,f"{height}_{width}_{num_bit}bit.pkl")
    with open(keys_path, 'rb') as f:  # Save the keys to a file
        encoding_key, decoding_key = pickle.load(f)
    
    video_pipe.scheduler = inverse_scheduler
    video_frames_dirs = os.path.join(args.video_frames_dir,"videomark",model_name,f"{num_bit}bit")
    json_output_path = os.path.join(video_frames_dirs, "temporal_results.json")
    results = load_json(json_output_path)
    for dirname in os.listdir(video_frames_dirs):

        video_frames_dir = os.path.join(video_frames_dirs, dirname,'wm','frames')
        shift_value = np.load(os.path.join(video_frames_dirs, dirname,"shift_value.npy"))

        if not os.path.exists(video_frames_dir):
            continue
    
        temporal_tampering_type = ['frame insert','frame drop','frame swap']
        video_frames_files = sorted(os.listdir(video_frames_dir))
        video_frames_files = [f'{video_frames_dir}/{file}' for file in video_frames_files]
        video_frames = [cv2.imread(frame_file) for frame_file in video_frames_files]
        video_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255. for frame in video_frames]

        # temporal tamper
        video_frames_tampered = temporal_tamper(video_frames, temporal_tampering_type, shift_value, message_bits_sequence)
        for tamper_type, (video_frames_t, frame_ids, message_t) in video_frames_tampered.items():
            print(f'Temporal tamper type: {tamper_type}')
            video_key = dirname            
            if video_key in results and (tamper_type in results[video_key]) and ('decode_acc' in results[video_key][tamper_type]) and ('frames_acc' in results[video_key][tamper_type]):
                print(f"Skipping {video_key}, results already exist.")
                continue
            else:
                results.setdefault(video_key, {})
                results[video_key].setdefault(tamper_type, {})

            tampered_num_frames = len(video_frames_t)
            first_frame = cv2_to_pil(video_frames_t[0])

            video_frames_t = transform_video(video_frames_t).to(video_pipe.vae.dtype).to(device)
            if model_name != 'stable-video-diffusion':
                with torch.no_grad():
                    video_latents = get_video_latents(video_pipe.vae, video_frames_t, sample=False, permute=True)
            else:
                with torch.no_grad():
                    video_latents = get_video_latents(video_pipe.vae, video_frames_t, sample=False, permute=False)

            if model_name == 'modelscope':
                reversed_latents = video_pipe(
                    prompt='',
                    latents=video_latents,
                    num_inference_steps=num_inversion_steps,
                    guidance_scale=1.,
                    output_type='latent',
                ).frames
            elif model_name == 'i2vgen-xl':
                reversed_latents = video_pipe(
                    prompt = 'None',
                    image=first_frame,
                    height=512,
                    width=512,
                    latents=video_latents.half(),
                    num_frames=tampered_num_frames,
                    output_type='latent',
                    num_inference_steps=50,
                    guidance_scale=6,
                ).frames
            else:
                raise ValueError

            frames_index, frames_message, frames_distance = process_frames_in_parallel(tampered_num_frames, reversed_latents, decoding_key, message_bits_sequence)
            decode_acc = np.mean(message_t[1:] == frames_message)
            frame_acc = np.mean(frame_ids[1:] == frames_index)
            print(f"decode_acc:{decode_acc}")
            print(f"frame_acc:{frame_acc}")
            results[video_key][tamper_type]['decode_acc'] = decode_acc
            results[video_key][tamper_type]['frames_acc'] = frame_acc
            save_json(results, json_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--model_name', default='i2vgen-xl')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--num_inversion_steps', default=50, type=int)
    parser.add_argument('--video_frames_dir', default="./results", type=str)
    parser.add_argument('--num_bit', default=512,type=int)
    parser.add_argument('--keys_path', default="./keys")
    args = parser.parse_args()
    main(args)
