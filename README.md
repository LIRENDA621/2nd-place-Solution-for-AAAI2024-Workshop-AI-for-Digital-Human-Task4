# 2nd-place-Solution-for-AAAI2024-Workshop-AI-for-Digital-Human-Task4



[[Challenge]](https://digitalhumanworkshop.github.io/)

**Please note that our code is built based on [[TalkSHOW]](https://github.com/yhw-yhw/TalkSHOW), [[SHOW]](https://github.com/yhw-yhw/SHOW), [[LSP]](https://github.com/YuanxunLu/LiveSpeechPortraits).**

## Setup environment
Clone the repo:
  ```bash
  git clone https://github.com/LIRENDA621/MMSports2023_-Player_Reidentification_Challenge.git
  cd MMSports2023_-Player_Reidentificatio_Challenge
  ```  
Create conda environment:
```bash
pip install -r requirements.txt
```

## Usage

Steps for Training and Evaluation:

1. get data: `download_data.py`
2. create DataFrames: `preprocess_data.py`
3. training:
   
   `train_our_post.py`
   
   The **main training script** will output two post-processing results during each verification, one is last year's championship solution, and the other is our implementation. The differences can be found in the paper.

   `train_our_post_resnet.py`

   This training script facilitates the use of the ResNet model to conduct relevant experiments in the paper.
   
5. evaluation: `evaluate.py`
   
7. final predictions:
   
    `predict.py`
   
   
   `predict_fortestset.py`
   
   
   The script is convenient for performing ablation experiments on the test set and does not need to be used.
