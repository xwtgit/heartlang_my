#!/bin/bash

export OMP_NUM_THREADS=1

PORT=${PORT:-2025}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

torchrun --nnodes=1 --master_port=${PORT} --nproc_per_node=1 run_class_finetuning.py \
    --dataset_dir datasets/ecg_datasets/DREAMER_QRS/va4 \
    --output_dir checkpoints/finetune/dreamer/finetune_va4_base_linear/ \
    --log_dir log/finetune/finetune_dreamer_va4_base_linear \
    --model HeartLang_finetune_base \
    --finetune checkpoints/heartlang_base/checkpoint-200.pth \
    --trainable linear \
    --task_type multiclass \
    --split_ratio 1 \
    --sampling_method random \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-3 \
    --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 100 \
    --layer_decay 0.9 \
    --save_ckpt_freq 100 \
    --seed 0 \
    --nb_classes 4 \
    --world_size 1
