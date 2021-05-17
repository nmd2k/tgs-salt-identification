import argparse

import torch
from torch._C import parse_ir
import torchvision

def train():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help="initial weights path")
    parser.add_argument('--data', type=str, default='./data', help="dataset path")
    parser.add_argument('--epochs', type=int, default=5, help="number of epoch")
    parser.add_argument('--batch-size', type=str, default='./data', help="total batch size for all GPUs")
    
    opt = parser.parse_args()