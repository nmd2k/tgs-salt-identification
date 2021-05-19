from numpy import deprecate_with_doc
import torch
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import matplotlib.pyplot as plt

class TGSDataset(Dataset):
    """TGS Salt Identification dataset."""
    
    def __init__(self, root_dir):
        """
        Args:
            root_path (string): Directory with all the images.
        """

        # load dataset from root dir
        train_df  = pd.read_csv(root_dir+'train.csv', index_col='id', usecols=[0])
        depths_df = pd.read_csv(root_dir+'depths.csv', index_col='id')
        train_df = train_df.join(depths_df)

        self.root_dir = root_dir
        self.ids      = train_df.index
        self.depths   = train_df['z'].to_numpy()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id    = self.ids[index]
        depth = self.depths[index]

        image = io.imread(self.root_dir+'train/images/'+id+'.png', as_gray=True)
        mask = io.imread(self.root_dir+'train/masks/'+id+'.png', as_gray=True)

        return (image, mask, depth)

def get_dataloader(data_path, batch_size, transformer, random_seed, valid_ratio=0.2, shuffle=True,num_workers=4):
    """
    Params:
    -------
    - data_path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - transformer: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
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

    # load the dataset
    orig_dataset = datasets.ImageFolder(root=data_path)

    # split the dataset
    n = len(orig_dataset)
    indices = list(range(n))
    n_valid = int(valid_ratio*n)

    train_idx, valid_idx = indices[n_valid:], indices[:n_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(orig_dataset, batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers)

    valid_loader = DataLoader(orig_dataset, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers)

    
def show_dataset(dataset, n_sample=4):
    """Visualize dataset with n_sample"""
    fig = plt.figure()

    # show image
    for i in range(n_sample):
        image, mask = dataset[i][:2]

        print(i, image.shape, mask.shape)

        plt.tight_layout()
        ax = plt.subplot(2, n_sample, i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        plt.imshow(image, cmap='gray')

    # show mask
    for i in range(n_sample):
        image, mask = dataset[i][:2]

        print(i, image.shape, mask.shape)

        ax = fig.add_subplot(2, n_sample, i + 1 + n_sample)
        plt.tight_layout()
        ax.set_title('Mask #{}'.format(i))
        ax.axis('off')

        plt.imshow(mask, cmap='gray')

        if i == n_sample-1:
            plt.show()
            break
