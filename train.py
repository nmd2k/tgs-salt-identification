import argparse
from model.loss import Weighted_Cross_Entropy_Loss
from model.config import BATCH_SIZE, DATA_PATH, EPOCHS, INPUT_SIZE, LEARNING_RATE, N_CLASSES

import torch
from tqdm import tqdm
from torchsummary import summary
from torch import optim, nn
import torch.nn.functional as F
from model.model import UNet, UNet_ResNet
from utils.dataset import TGSDataset, get_dataloader, get_transform, show_dataset, show_image_mask

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
    running_loss = 0
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
        running_loss += loss.item()

    total_loss = running_loss/len(trainloader.dataset)
    return total_loss
    
def test(model, device, testloader, loss_function):
    model.eval()
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for idx, (input, mask) in enumerate(testloader):
            input, mask = input.to(device), mask.to(device)

            predict = model(input)
            loss = loss_function(predict, mask)
            # test_loss += F.nll_loss(predict, mask, size_average=False).item()
            # output = predict.data.max(1, keepdim=True)[1]
            # correct += output.eq(mask.view_as(output)).sum().item()
    
    test_loss /= len(testloader)
    # test_accuracy = 100. * correct / len(testloader.dataset)
    
    return test_loss

if __name__ == '__main__':
    args = parse_args()

    # train on device
    device = {"cuda:0" if torch.cuda.is_available() else "cpu"}

    # load dataset
    transform = get_transform()
    dataset = TGSDataset(DATA_PATH, transforms=transform)
    trainloader, validloader = get_dataloader(dataset=dataset)

    # get model and define loss func, optimizer
    n_classes = N_CLASSES
    model = UNet()
    epochs = EPOCHS

    # summary model
    summary = summary(model, input_size=(3, INPUT_SIZE, INPUT_SIZE))

    criterion = nn.BCEWithLogitsLoss()

    # loss_func   = Weighted_Cross_Entropy_Loss()
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    pb = tqdm(range(epochs))
    train_losses, test_losses, test_accuracy = [], [], []

    for epoch in pb:
        train_loss = train(model, device, trainloader, optimizer, criterion)
        train_losses.append(train_loss)

        test_loss = train(model, device, validloader, criterion)
        test_losses.append(test_loss)
        # test_accuracy.append(test_acc)

        pb.set_description(f'Train loss: {train_loss} | Valid loss: {test_loss}')