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

### Data preprocess

Generate data for TalkSHOW and LSP

```bash
cd SHOW/SHOW
sh multi_demo.sh
```

### Audio2gesture
    
    Train VQ-VAEs. 
    bash my_train_body_vq.sh
    # 2. Train PixelCNN. Please modify "Model:vq_path" in config/body_pixel.json to the path of VQ-VAEs.
    bash my_train_body_pixel.sh
    # 3. Train face generator.
    bash my_train_face.sh


