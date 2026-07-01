#!/usr/bin/env bash
# Evaluate LoRA-fine-tuned SAM-Audio on LASS evaluation splits.
# Run after: python lora_train.py
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python lora_train.py > lora_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python turbo_train.py --num-steps 4 --epochs 20 --use-lora > turbo_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python eval/main.py \
    --setting lass-synth lass-real \
    --checkpoint-path "lass-lora-merged" \
    --batch-size 1 \
    --metrics clap aes judge > eval.log 2>&1 &