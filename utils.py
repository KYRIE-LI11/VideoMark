import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from transformers import get_cosine_schedule_with_warmup
import os
import json

def cv2_to_pil(img):
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    return pil_img


def pil_to_cv2(pil_img):
    img = np.asarray(pil_img) / 255
    return img


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs

@torch.inference_mode()
def get_video_latents(vae, video_frames, sample=True, rng_generator=None, permute=True):
    encoding_dist = vae.encode(video_frames).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = (encoding * 0.18215).unsqueeze(0)
    if permute:
        latents = latents.permute(0, 2, 1, 3, 4)
    return latents

@torch.inference_mode()
def latents_to_video(pipe, latents, num_frames=None):
    if num_frames is None:
        video_tensor = pipe.decode_latents(latents)
    else:
        video_tensor = pipe.decode_latents(latents, num_frames)
    video = tensor2vid(video_tensor, pipe.video_processor)
    return video


def save_video_frames(frames, save_dir):
    if isinstance(frames[0], np.ndarray):
        frames = [(frame * 255).astype(np.uint8) for frame in frames]

    elif isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]

    for i in range(len(frames)):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{save_dir}/{i:02d}.png',img)


def transform_video(video):
    tform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    video_tensor = torch.stack([2.0 * tform(frame) - 1.0 for frame in video])
    return video_tensor


def recover(idx_list, message_list, distance_list, num_bit):#recover the original message 

    valid_entries = [(idx, msg, dist) for idx, msg, dist in zip(idx_list, message_list, distance_list) if idx != -1]
    valid_entries.sort(key=lambda x: x[0])
    
    sorted_order = []
    recovered_message = []
    recovered_distance = []
    
    index = 0
    for idx in idx_list:
        if idx == -1:
            sorted_order.append(-1)
            recovered_message.append([-1]*num_bit)
            recovered_distance.append(num_bit)
        else:
            sorted_order.append(valid_entries[index][0])
            recovered_message.append(valid_entries[index][1])
            recovered_distance.append(valid_entries[index][2])
            index += 1
    return np.array(sorted_order), np.array(recovered_message), np.array(recovered_distance)