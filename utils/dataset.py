from model.config import *
from numpy import deprecate_with_doc
import torch
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader, dataloader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

class TGSDataset(Dataset):
    """TGS Salt Identification dataset."""
    
    def __init__(self, root_dir=DATA_PATH, transform=None):
        """
        Args:
            root_path (string): Directory with all the images.
            transformer (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
        """

        # load dataset from root dir
        train_df  = pd.read_csv(root_dir+'train.csv', index_col='id')
        depths_df = pd.read_csv(root_dir+'depths.csv', index_col='id')
        train_df = train_df.join(depths_df)

        self.root_dir   = root_dir
        self.ids        = train_df.index
        self.depths     = train_df['z'].to_numpy()
        self.rle        = train_df['rle_mask'].to_numpy()
        self.transfroms = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id    = self.ids[index]
        depth = self.depths[index]

        # file should be unzipped
        image = io.imread(self.root_dir+IMAGE_PATH+id+'.png', as_gray=True)
        mask  = io.imread(self.root_dir+MASK_PATH+id+'.png', as_gray=True)

        if self.transfroms:
            image = self.transfroms(image)
            mask = self.transfroms(mask)

        return (depth, image, mask)

def get_transform(is_train=True):
    transform = [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]

    if is_train:
        transform += [
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            )],p=0.6),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=0.1
            )],p=0.6),
        ]

    transform += [
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ]

    return transforms.Compose(transforms)


def get_dataloader(dataset, 
                    batch_size=BATCH_SIZE, random_seed=RANDOM_SEED, 
                    valid_ratio=VALID_RATIO, shuffle=True, num_workers=NUM_WORKERS):
    """
    Params:
    -------
    - dataset: the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_ratio: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    
    Returns:
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    error_msg = "[!] valid_ratio should be in the range [0, 1]."
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), error_msg

    # split the dataset
    n = len(dataset)
    n_valid = int(valid_ratio*n)
    n_train = n - n_valid

    # indices = list(range(n))
    # train_idx, valid_idx = indices[n_valid:], indices[:n_valid]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # init random seed
    torch.manual_seed(random_seed)

    train_dataset, valid_dataset = random_split(dataset, (n_train, n_valid))

    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
    #                 num_workers=num_workers)

    # valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
    #                 num_workers=num_workers)

    train_loader = dataloader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_loader = dataloader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader
    
def show_dataset(dataset, n_sample=4):
    """Visualize dataset with n_sample"""
    fig = plt.figure()

    # show image
    for i in range(n_sample):
        image, mask = dataset[i][2:]

        print(i, image.shape, mask.shape)

        plt.tight_layout()
        ax = plt.subplot(2, n_sample, i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        plt.imshow(image, cmap='gray')

    # show mask
    for i in range(n_sample):
        image, mask = dataset[i][2:]

        print(i, image.shape, mask.shape)

        ax = fig.add_subplot(2, n_sample, i + 1 + n_sample)
        plt.tight_layout()
        ax.set_title('Mask #{}'.format(i))
        ax.axis('off')

        plt.imshow(mask, cmap='gray')

        if i == n_sample-1:
            plt.show()
            break
