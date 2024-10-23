#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

module load cuda cudnn anaconda
source activate lint
conda activate hllm2




CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset Pixel200K \
    --train_batch_size 1 \
    --MAX_TEXT_LENGTH 16 \
    --MAX_ITEM_LIST_LENGTH 1 \
    --checkpoint_dir ./ \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir /home/li4256/HLLM/tinyllama \
    --user_pretrain_dir /home/li4256/HLLM/tinyllama \
    --text_path /home/li4256/HLLM/dataset/information/ \
    --text_keys '[\"title\",\"description\"]' \
    --val_only True
