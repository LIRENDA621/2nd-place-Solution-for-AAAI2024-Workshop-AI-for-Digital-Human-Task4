#!/bin/bash

data_dir=data/personA_train_val
# resuts=results/person_A_padding

# export CUDA_VISIBLE_DEVICES=3

python3 test_feature2face_model_demo.py \
          --name personA_train_val \
          --dataset_names 'personA_train_val' \
          --test_dataset_names 'personA_train_val' \
          --dataroot $data_dir \
          --isH5 0 \
          --checkpoints_dir checkpoints/personB_train_val_mask1 \
          --dataset_mode customtestdemo \
          --results_dir results/personA_train_val \
          --gpu_ids 2 \
          --load_epoch checkpoints/personA_train_val_mask1/personA/50_Feature2Face_G.pkl
