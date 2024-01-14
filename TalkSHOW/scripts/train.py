import os
import sys
# os.chdir('/home/jovyan/Co-Speech-Motion-Generation/src')
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(os.getcwd())
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()