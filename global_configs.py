import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")
MOSI_ACOUSTIC_DIM = 74
MOSI_VISUAL_DIM = 47

ACOUSTIC_DIM = 74
VISUAL_DIM = 47

TEXT_DIM = 768
