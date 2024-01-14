#!/bin/bash

data_dir=data/personA_train
name=person_A_train_full_mask2

# export CUDA_VISIBLE_DEVICES=3

python3 train_feature2face_model.py \
            --name person_A \
            --dataset_names $name \
            --train_dataset_names 'person_A_train_full_mask2' \
            --dataroot $data_dir \
            --isH5 0 \
            --checkpoints_dir checkpoints/$name \
            --dataset_mode custom \
            --gpu_ids 3
