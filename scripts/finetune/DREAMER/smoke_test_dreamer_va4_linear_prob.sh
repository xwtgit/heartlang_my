#!/bin/bash

export OMP_NUM_THREADS=1

PORT=${PORT:-2026}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

torchrun --nnodes=1 --master_port=${PORT} --nproc_per_node=1 run_class_finetuning.py \
    --dataset_dir datasets/ecg_datasets/DREAMER_QRS/va4 \
    --output_dir checkpoints/finetune/dreamer/smoke_test_10s/ \
    --log_dir log/finetune/dreamer_smoke_test_10s \
    --model HeartLang_finetune_base \
    --finetune checkpoints/heartlang_base/checkpoint-200.pth \
    --trainable linear \
    --task_type multiclass \
    --batch_size 8 \
    --epochs 1 \
    --warmup_epochs 0 \
    --nb_classes 4 \
    --world_size 1 \
    --no_auto_resume
