from model.metric import cal_iou
from model.config import DATA_PATH
from model.config import *
from PIL import Image 
from torchvision import transforms

id = '00a3af90ab'
image = Image.open(DATA_PATH+IMAGE_PATH+id+'.png').convert('L')
mask  = Image.open(DATA_PATH+MASK_PATH+id+'.png').convert('L')

transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor()])

image = transforms(image)
mask = transforms(mask)

predict = mask

print(cal_iou(predict, mask))