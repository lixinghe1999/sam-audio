CUDA_VISIBLE_DEVICES=0 nohup python turbo_train.py \
  --objective meanflow \
  --use-lora \
  --epochs 20 > meanflow_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python turbo_train.py \
  --objective consistency_distillation \
  --use-lora \
  --num-steps 4 > cd_train.log 2>&1 &