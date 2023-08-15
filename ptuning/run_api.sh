#!/bin/bash

# 设置默认值
default_pre_seq_len=64
default_lr=1e-4
default_base_path=/data
default_quantization_bit=8
default_max_steps=3000

# 通过环境变量或默认值来赋值
pre_seq_len=${PRE_SEQ_LEN:-$default_pre_seq_len}
lr=${LR:-$default_lr}
base_path=${BASE_PATH:-$default_base_path}
quantization_bit=${QUANTIZATION_BIT:-$default_quantization_bit}
max_steps=${MAX_STEPS:-$default_max_steps}

echo "pre_seq_len: $pre_seq_len"
echo "lr: $lr"
echo "base_path: $base_path"
echo "quantization_bit: $quantization_bit"
echo "max_steps: $max_steps"

CUDA_VISIBLE_DEVICES=0 python3 api.py \
    --model_name_or_path "$base_path"/models/chatglm-6b \
    --ptuning_checkpoint "$base_path"/output/adgen-chatglm-6b-pt-"$pre_seq_len"-"$lr"-"$quantization_bit"/checkpoint-"$max_steps" \
    --pre_seq_len "$pre_seq_len" \
    --prefix_projection \
    --quantization_bit "$quantization_bit"