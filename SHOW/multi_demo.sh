#!/bin/bash

folder_path="/data/user/lirenda621/Data/pats/noah_vad_3500_512_288_undealed_1"

for mp4_file in "$folder_path"/*/*.mp4; do
# for mp4_file in "$folder_path"/*.mp4; do

    if [ -e "$mp4_file" ]; then
        echo "Processing file: $mp4_file"
        export mmpose_root="third_party/mmpose-0.29.0"
        export CUDA_VISIBLE_DEVICES=0
        python main.py --speaker_name "-1" --all_top_dir "$mp4_file" --fps "30"
    fi
done


