# Consistency-StableDiffusion(WIP)
Consistency Models for Stable-Diffusion(CSD)

This is an experimental implementation of Consistency Training with StableDiffusion. The purpose is to achieve one-step sampling by training/finetuning/distilling a Stable-Diffusion using Songyang's Consistency Models approach. 

This codebase is built on top of the [openai/consistency_models](https://github.com/openai/consistency_models) and HuggingFace [diffusers](https://github.com/huggingface/diffusers) libraries.

Since Original Consistency models were based on EDM, Stable-Diffusion was based on DDPM and Latent Diffusion. So I implement Consistency-SD based on my personal understanding. Any contributions/suggestions/feedback are welcomed!

## Training
**Because of the high cost of training and distilling CSD, I first try to finetune a CSD directly on the SD base to see if it is feasible.**

Training can be launched with the following bash command,

```bash
cd examples/text_to_image/

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_consistency.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" \
  --report_to wandb \
  --validation_prompts "a cute pikachu", "yoda"

```

I am currently training with this script, using 5-step Consistency Sampling. The intermediate results(8500 steps) seem to be reasonable. 

|    |    |    | 
|----|----|----|
| ![Cute Obama creature](https://user-images.githubusercontent.com/22982797/233788048-8fa2651e-9600-4a6d-800b-6ad6247c4bf8.png) | ![Hello Kitty_3](https://user-images.githubusercontent.com/22982797/233788050-5a468235-b389-4b06-8474-023cd7d13cbc.png) | ![Totoro_3](https://user-images.githubusercontent.com/22982797/233788051-b6f9f58e-6459-487b-a451-b0275700b264.png)|

I believe further training and below ``TODO`` items may improve the image quality. 

## Inference
Images can be sampled using codes

```python
from diffusers import StableDiffusionConsistencyPipeline

pipe = StableDiffusionConsistencyPipeline.from_pretrained("yangyfaker/sd-consistency-exp-pokemon")
pipe.to("cuda")
prompts = ["Cute Obama creature", "Totoro", "Hello Kitty"]
for prompt in prompts:
    image = pipe(prompt=prompt, num_inference_steps=10).images[0]
    image.save(f"{prompt}.png")
```

## TODO

- [ ] set N(k)
- [ ] apply text dropout
- [ ] support distillization
- [ ] align loss reweighting
- [ ] select loss type: l1 and lpips
- [ ] align/adjust unet ema setting
- [ ] align/adjust learning rate && Optimizer
