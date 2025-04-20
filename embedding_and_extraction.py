import os
import numpy as np
import json
import argparse
from tqdm import tqdm
import torch
import pickle
from src.prc import Encode, Detect, Decode
import src.pseudogaussians as prc_gaussians
from diffusers import TextToVideoSDPipeline, I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import DDIMInverseScheduler
import galois
import multiprocessing
from Levenshtein import distance
GF = galois.GF(2)
from utils import (
    transform_video,
    save_video_frames,
    cv2_to_pil,
    get_video_latents,
    recover,
)
import numpy as np
np.random.seed(11111)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def bits_to_string(bits):
    return ''.join(map(str, bits))

def process_frame(frame_index, reversed_latents_w_cpu, decoding_key, message_bits):
    reversed_prc = prc_gaussians.recover_posteriors(
        reversed_latents_w_cpu[:,:,frame_index].to(torch.float64).flatten(), 
        variances=1.5
    ).flatten()
    
    detection_result = Detect(decoding_key, reversed_prc, false_positive_rate=0.01)
    message_placeholder = '<message_placeholder>'
    if not detection_result:
        decode_message_str = message_placeholder
    else:
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
                                for frame_index in range(0,num_frames)])
    
    for decode_message, distance, idx in results:
        message_list.append(decode_message)
        distance_list.append(distance)
        idx_list.append(idx)

    recovered_index, recovered_message, recovered_distance = recover(idx_list, message_list, distance_list, num_bit)
    return recovered_index, recovered_message, recovered_distance

    
def get_message_bits(message_bits, length=16):
    num_rows = message_bits.shape[0]
    shift = np.random.RandomState().randint(0, num_rows-length)
    return shift, message_bits[shift:shift+length]

def main(args):
    device = args.device
    model_name = args.model_name
    data_dir = args.data_dir
    use_watermark = args.use_watermark
    height = args.height
    width = args.width
    num_bit = args.num_bit

    keys_path = os.path.join(args.keys_path,f"{height}_{width}_{num_bit}bit.pkl")
    message_bits_sequence = np.random.randint(0, 2, size=(500,num_bit))
    
    if use_watermark:
        output_dir = os.path.join(args.output_dir, "videomark", model_name, f"{num_bit}bit")
    else:
        output_dir = os.path.join(args.output_dir, "without_watermark", model_name)
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "video_results.json")

    assert model_name is not None, 'you must provide the model name!'

    if args.model_path is not None:
        model_path = args.model_path
    else:
        if model_name == 'modelscope':
            model_path = 'damo-vilab/text-to-video-ms-1.7b'
        elif model_name == 'i2vgen-xl':
            model_path = 'ali-vilab/i2vgen-xl'
        else:
            raise ValueError

    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        num_frames=args.num_frames
    elif model_name == 'i2vgen-xl':
        video_pipe = I2VGenXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)

        num_frames=args.num_frames
    else:
        raise ValueError

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        model_path,
        subfolder='scheduler',
    )
    video_pipe.safety_checker = None
    scheduler = video_pipe.scheduler


    with open(f'./{data_dir}/test_prompts.txt', 'r') as f:
        data = [line.strip() for line in f.readlines()]

    args_dict = vars(args)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4) 

    bit_acc_list = []
    results = load_json(json_output_path)

    for item in tqdm(range(4)):
        for i, row in tqdm(enumerate(data)):
            current_prompt = row

            video_id = current_prompt.replace(' ', '_')

            video_key = f"{video_id}_{item}"

            if video_key in results:
                print(f"Skipping {video_key}, results already exist.")
                continue
            else:
                results.setdefault(video_key, {})
                
            video_dir = os.path.join(output_dir, video_key)
            os.makedirs(video_dir, exist_ok=True)

            print(f'Item: {item}\nGenerating for prompt: {current_prompt}.')

            if use_watermark:
                with open(keys_path, 'rb') as f:
                    encoding_key, decoding_key = pickle.load(f)
                shift, message_bits = get_message_bits(message_bits_sequence)

                prc_codewords = torch.stack([Encode(encoding_key, message=message_bits[i]).to(device) for i in range(num_frames)])
                latents = torch.randn_like(prc_codewords).to(device)
                frame_latents = (prc_codewords * torch.abs(latents)).reshape(num_frames, 1, 4, height, width).to(dtype=torch.float16)
                init_latents_w = frame_latents.permute(1, 2, 0, 3, 4)

            else:
                init_latents_w = torch.randn(num_frames, 1, 4, height, width).to(device)

            print(f'Video generation and watermark embedding:')
            video_pipe.scheduler = scheduler
            retries = 0
            if model_name == 'modelscope':
                video_frames_w = video_pipe(
                    prompt=current_prompt,
                    latents=init_latents_w,
                    num_frames=args.num_frames,
                    height=int(height*8),
                    width=int(width*8),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=9.0,
                ).frames[0]
            elif model_name == 'i2vgen-xl':
                image_path = f'./{data_dir}/img_prompt/{video_id}_img{item}.png'
                image = load_image(image_path)

                video_frames_w = video_pipe(
                    image=image,
                    prompt = current_prompt,
                    height=int(height*8),
                    width=int(width*8),
                    num_frames=args.num_frames,
                    decode_chunk_size=16,
                    latents=init_latents_w,
                    output_type='np',
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=9,
                ).frames[0]
            else:
                raise ValueError

            save_mp4_path = os.path.join(video_dir, f'wm.mp4')
            np.save(os.path.join(video_dir,'shift_value.npy'), shift)
            export_to_video(video_frames_w, output_video_path=save_mp4_path)
            print(f'The generated video is saved to {save_mp4_path}.')
            frames_dir = f'{video_dir}/wm/frames'
            os.makedirs(frames_dir, exist_ok=True)
            save_video_frames(video_frames_w, frames_dir)
            print(f'The generated video frames are saved to {frames_dir}.')

            first_frame = cv2_to_pil(video_frames_w[0])
            video_frames_w = transform_video(video_frames_w).to(video_pipe.vae.dtype).to(device)

            video_latents_w = get_video_latents(video_pipe.vae, video_frames_w, sample=False, permute=True)

            print(f'Watermark extraction:')
            video_pipe.scheduler = inverse_scheduler
            if model_name == 'modelscope':
                # assume at the detection time, the original prompt is unknown
                reversed_latents_w = video_pipe(
                    prompt='',
                    latents=video_latents_w,
                    num_inference_steps=args.num_inversion_steps,
                    guidance_scale=1.,
                    output_type='latent',
                ).frames
            elif model_name == 'i2vgen-xl':
                reversed_latents_w = video_pipe(
                    prompt = 'None',
                    image=first_frame,
                    height=512,
                    width=512,
                    latents=video_latents_w,
                    num_frames=args.num_frames,
                    output_type='latent',
                    num_inference_steps=args.num_inversion_steps,
                    guidance_scale=9,
                ).frames
            else:
                raise ValueError

            frames_index, frames_message, frames_distance = process_frames_in_parallel(num_frames, reversed_latents_w, decoding_key, message_bits_sequence)

            bit_acc = np.mean(frames_message[1:]==message_bits[1:])
            bit_acc_list.append(bit_acc)
            
            print(f"decode acc:{bit_acc}")
            results[video_key] = {
                "model": model_name,
                "decode_acc": bit_acc,
            }
            save_json(results, json_output_path)
    decode_acc_values = [entry["decode_acc"] for entry in results.values() if entry["decode_acc"] is not None]

    avg_decode_acc = sum(decode_acc_values) / len(decode_acc_values) if decode_acc_values else 0

    print(f"Average Decode Accuracy: {avg_decode_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoShield')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--height', default=64, type=int)
    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--model_name', default='i2vgen-xl')
    parser.add_argument('--use_watermark', default=True)
    parser.add_argument('--keys_path', default="./keys")
    parser.add_argument('--num_bit', default=512,type=int)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)


