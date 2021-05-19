import argparse

import torch
import torchvision
from utils.dataset import TGSDataset, show_dataset

import matplotlib.pyplot as plt

def train():
    # define transform function

    # load dataset
    dataset = TGSDataset('./data/')
    
    show_dataset(dataset)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help="initial weights path")
    parser.add_argument('--data', type=str, default='./data', help="dataset path")
    parser.add_argument('--epochs', type=int, default=5, help="number of epoch")
    parser.add_argument('--batch-size', type=str, default='./data', help="total batch size for all GPUs")
    
    opt = parser.parse_args()
    train()