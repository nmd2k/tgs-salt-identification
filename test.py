import os
import pandas as pd
from PIL import Image
from skimage import io

root_dir = './data/'

train_df  = pd.read_csv(root_dir+'train.csv', index_col='id')
depths_df = pd.read_csv(root_dir+'depths.csv', index_col='id')
train_df = train_df.join(depths_df)

images = {}
path = root_dir+'train/images/'
filename = os.listdir(path)[:3]

# images['id'] = ['images', 'masks']
image_df = pd.DataFrame()

for file in filename:
    id = file[:-4]
    img_array = io.imread(path+file, as_gray=True)
    image_df = image_df.append({"id": id, "image": img_array}, ignore_index=True)

# print(image_df.set_index('id'))

train_df = train_df.join(image_df.set_index('id'))

ids = train_df.index
depths = train_df['z'].to_numpy()
print(depths)