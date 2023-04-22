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

I am currently training with this script, using 5-step Consistency Sampling. The intermediate results seem to be reasonable. 

## TODO
- [ ] set N(k)
- [ ] support distillization
- [ ] align loss reweighting
- [ ] select loss type: l1 and lpips?
- [ ] align/adjust unet ema setting
- [ ] align/adjust learning rate && Optimizer
