import argparse
from utils.utils import wandb_mask

from torchvision import transforms
from model.metric import cal_iou
import os
from traceback import walk_tb

from torch.utils import data
from model.loss import Weighted_Cross_Entropy_Loss
from model.config import BATCH_SIZE, DATA_PATH, EPOCHS, INPUT_SIZE, LEARNING_RATE, N_CLASSES, RUN_NAME, SAVE_PATH, START_FRAME

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from torch import optim, nn
import torch.nn.functional as F
from model.model import UNet, UNet_ResNet
from utils.dataset import TGSDataset, get_dataloader, get_transform
from utils.utils import show_dataset, show_image_mask

import matplotlib.pyplot as plt

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--weights', type=str, default='', help="initial weights path")
    parser.add_argument('--data', type=str, default=DATA_PATH, help="dataset path")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--batch-size', type=str, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="learning rate (default: 0.0001)")
    args = parser.parse_args()
    return args

def train(model, device, trainloader, optimizer, loss_function):
    model.train()
    running_loss = []
    running_iou = []
    for i, (input, mask) in enumerate(trainloader):
        # load data into cuda
        input, mask = input.to(device), mask.to(device)

        # zero the gradient
        optimizer.zero_grad()

        # forward + backpropagation + step
        predict = model(input)
        loss = loss_function(predict, mask)
        
        loss.backward()
        optimizer.step()

        # statistics
        running_iou.append(cal_iou(predict, mask))
        running_loss.append(loss.item())

        # Report metrics every 25th batch
        if ((i + 1) % 25) == 0:
            temp_loss = np.mean(running_loss)
            temp_iou  = np.mean(running_iou)
            wandb.log({'Train loss': temp_loss, 'Train IoU': temp_iou}, step=i+1)
            pb.set_description(f'Train loss: {train_loss} | Train IoU: {train_iou}', refresh=False)
            
    total_loss = np.mean(running_loss)
    mean_iou = np.mean(running_iou)

    return total_loss, mean_iou
    
def test(model, device, testloader, loss_function):
    model.eval()
    running_loss = []
    running_iou  = []
    mask_list    = []
    with torch.no_grad():
        for i, (input, mask) in enumerate(testloader):
            input, mask = input.to(device), mask.to(device)

            predict = model(input)
            loss = loss_function(predict, mask)

            running_loss.append(loss.item())
            running_iou.append(cal_iou(predict, mask))

            # log the first image of the batch
            bg_img = input.cpu().detach().numpy()
            true_mask = mask.cpu().detach().numpy()
            prediction_mask = predict.cpu().detach().numpy()
            mask_list.append(wandb_mask(bg_img, prediction_mask, true_mask))
            
            # Report metrics every 25th batch
            if ((i + 1) % 25) == 0:
                temp_loss = np.mean(running_loss)
                temp_iou  = np.mean(running_iou)
                wandb.log({'Test loss': temp_loss, 'Test IoU': temp_iou}, step=i+1)
                pb.set_description(f'Test loss: {train_loss} | Test IoU: {train_iou}', refresh=False)

    test_loss = np.mean(running_loss)
    mean_iou = np.mean(running_iou)

    return test_loss, mean_iou

if __name__ == '__main__':
    args = parse_args()

    # train on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("current device", device)

    # init wandb
    config = dict(
        lr          = LEARNING_RATE,
        batchsize   = BATCH_SIZE,
        epoch       = EPOCHS,
        adam        = True,
        model_sf    = START_FRAME,
        device      = device,
        data        = DATA_PATH
    )

    run = wandb.init(project="TGS-Salt-identification", tags=['Unet'], config=config)
    artifact = wandb.Artifact('tgs-salt', type='dataset')

    try:
        for dir in ['train', 'test']:
            artifact.add_dir(DATA_PATH+dir)
        for file in ['train.csv', 'depths.csv']:
            artifact.add_file(DATA_PATH+file)
    except:
        artifact     = run.use_artifact('tgs-salt:latest')
        artifact_dir = artifact.download(DATA_PATH)

    run.log_artifact(artifact)

    # load dataset
    transform = get_transform()
    dataset = TGSDataset(DATA_PATH, transforms=transform)
    trainloader, validloader = get_dataloader(dataset=dataset)

    # get model and define loss func, optimizer
    n_classes = N_CLASSES
    model = UNet().to(device)
    epochs = EPOCHS

    # summary model
    summary = summary(model, input_size=(1, INPUT_SIZE, INPUT_SIZE))

    criterion = nn.BCEWithLogitsLoss()

    # loss_func   = Weighted_Cross_Entropy_Loss()
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # wandb watch
    run.watch(models=model, criterion=criterion, log='all', log_freq=10)

    # training
    pb = tqdm(range(epochs))
    best_iou = -1

    for epoch in pb:
        train_loss, train_iou = train(model, device, trainloader, optimizer, criterion)

        test_loss, test_iou = test(model, device, validloader, criterion)

        pb.set_description(f'Epoch: {epoch} | Train loss: {train_loss} | Train IoU: {train_iou} | Valid loss: {test_loss} | Valid IoU: {test_iou}', refresh=False)
        
        # Wandb summary
        if best_iou < test_iou:
            best_iou = test_iou
            wandb.run.summary["best_accuracy"] = best_iou

    # saving model
    print("Train finished. Start saving model")

    # export to onnx + pt
    # torch.onnx.export(model, input, SAVE_PATH+RUN_NAME+'.onnx')
    torch.save(model, SAVE_PATH+RUN_NAME+'.pth')
    
    trained_weight = wandb.Artifact(RUN_NAME, type='weights')
    trained_weight.add_file(SAVE_PATH+RUN_NAME+'.onnx')
    trained_weight.add_file(SAVE_PATH+RUN_NAME+'.pth')
    run.log_artifact(trained_weight)

    # evaluate