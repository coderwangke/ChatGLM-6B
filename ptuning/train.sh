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

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file "$base_path"/datasets/train.json \
    --validation_file "$base_path"/datasets/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path "$base_path"/models/chatglm-6b \
    --output_dir "$base_path"/output/adgen-chatglm-6b-pt-"$pre_seq_len"-"$lr"-"$quantization_bit" \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps "$max_steps" \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate "$lr" \
    --pre_seq_len "$pre_seq_len" \
    --prefix_projection \
    --quantization_bit "$quantization_bit"