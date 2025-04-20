import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm

def generate_images_from_prompts(
    prompt_file: str,
    output_dir: str,
    model_id: str = "stabilityai/stable-diffusion-2-1",
    items_num: int = 4,
    height: int = 512,
    width: int = 512,
    device: str = "cuda"
):
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        safe_prompt = prompt.replace(' ', '_').replace('/', '_')
        print(f"Generating for prompt: {prompt}")
        
        images = pipe(prompt, height=height, width=width, num_images_per_prompt=items_num).images
        for j, image in enumerate(images):
            image_path = os.path.join(output_dir, f"{safe_prompt}_img{j+1}.png")
            image.save(image_path)
    print(f"Done! Images saved to: {output_dir}")



if __name__ == "__main__":
    generate_images_from_prompts(
        prompt_file="test_prompts.txt",
        output_dir="./img_prompt",
        items_num=1,
        height=512,
        width=512
    )
