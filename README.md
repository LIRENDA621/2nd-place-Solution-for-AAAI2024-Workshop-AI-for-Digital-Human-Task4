# 2nd-place-Solution-for-AAAI2024-Workshop-AI-for-Digital-Human-Task4



[[Challenge]](https://digitalhumanworkshop.github.io/)

**Please note that our code is built based on [[TalkSHOW]](https://github.com/yhw-yhw/TalkSHOW), [[SHOW]](https://github.com/yhw-yhw/SHOW), [[LSP]](https://github.com/YuanxunLu/LiveSpeechPortraits).**

## Setup environment
**Clone the repo**:
  ```bash
  git clone https://github.com/LIRENDA621/2nd-place-Solution-for-AAAI2024-Workshop-AI-for-Digital-Human-Task4.git
  ```

**For TalkSHOW**

Create conda environment:
```bash
conda create --name talkshow python=3.7
conda activate talkshow
```
Please install pytorch (v1.10.1).

    pip install -r requirements.txt
    
Please install [**MPI-Mesh**](https://github.com/MPI-IS/mesh).

**For LSP**

```bash
pip install -r requirements.txt
```

**For SHOW & OpenPose**

Environmental dependencies are complex, please refer to [[SHOW]](https://github.com/yhw-yhw/SHOW)


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
