from model.config import INPUT_SIZE
from os import name
from numpy.lib.type_check import imag
from torchvision.transforms.transforms import Scale
from model.loss import DiceLoss
from model.metric import cal_iou
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch import nn
from model.model import UNet, UNet_ResNet

loss_func = DiceLoss()


trans = transforms.Compose([transforms.Scale((INPUT_SIZE, INPUT_SIZE)), transforms.Grayscale(), transforms.ToTensor(),])
image = Image.open('data/train/images/0a1742c740'+'.png')
mask = Image.open('data/train/masks/0a1742c740'+'.png')
# mask = np.array(mask)//255

# print(mask)
image = trans(image)
mask = trans(mask)

mask = mask.unsqueeze(0)
image = image.unsqueeze(0)

print('input', image.shape)

net = UNet_ResNet()

predict = net(image)

# random1 = torch.randint(0, 2, (5, 5)).unsqueeze(0)
# random2 = torch.randint(0, 2, (5, 5)).unsqueeze(0)


# random = torch.randint(0, 2, (101, 101)).unsqueeze(0)

# print(random.size())
# loss = loss_func(random, mask)
# mask = trans(mask)
# print(mask.shape)
# print(random1.shape)
# print(random2.shape)

# loss = loss_func(random1, random2)

# print(loss.item())

# print(mask)
# print(random)

# print(cal_iou(random, mask))