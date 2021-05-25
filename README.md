# TGS Salt Identification

# [Abstract](#abstract)
**TGS Salt Identification** is a Kaggle Competition announced in 2018. 
The competition focus on the task of salt identification through seismic image. Seismic image is collected using reflection seismology, or seismic reflection. Take these images as example, where the red overlay region is refered to the salt region in this area:
![seismic image](imgs/sample.png)

**Data**
The data is a set of images chosen at various locations chosen at random in the subsurface, which contains 4000 images in the training dataset and 18000 images in the test set. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment.

# [Training & Result](#res)

**For training**: We splited the dataset into 2 parts, 1 holds **80%** of the datset will be used for `training` and other **20%** will served for `validation`.

We implement a custom dataset for loading TGS salt data into model. That is noticable that TGS data have 1 feature named `depths` which we didn't use for training, however, we still believe that it will bring some improvement to your model. Therefore, we still load the `depths.csv` into our datset in order to serve your later usages.
```python
class TGSDataset(Dataset):
    """TGS Salt Identification dataset."""
    
    def __init__(self, root_dir=DATA_PATH, transform=None):
        # load dataset from root dir
        train_df  = pd.read_csv(root_dir+'train.csv', index_col='id')
        depths_df = pd.read_csv(root_dir+'depths.csv', index_col='id')
        train_df = train_df.join(depths_df)

        self.depths     = train_df['z'].to_numpy()
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        id    = self.ids[index]
        depth = self.depths[index]
        ...
```
We implemented 2 models located in `model.py`, one is the original Unet [[1]](#1), the other was Unet based ResNet [[2]](#2). The Unet based ResNet was inspired by residual block architecture, in there, we attempt to introduce some new skip connection to the Unet architecture. The architecture of the Unet based Resnet is shown below:

![Unet resnet](imgs/Unet_Resnet.png)

However, you might experiment some downside while using this architecture than the original. We are still researching deeper into this Unet based Resnet model and the process is not finished yet. Therefore, use at your own risk.

**Result**: Due to the lack of computational power, we were using `Colab GPU` to train our model. After train 10 epoch each model with the dataset's `batch size` is 16 and the `start frame` is 64, the result are summary in the table below:

|             | Start frame | Batch size | Learning rate | Dropout rate |   | IoU<sup>train | IoU<sup>val |
|-------------|-------------|------------|---------------|--------------|---|-----------|----------|
| Unet        | 64          | 16         | 0.00017       | -            |   | 70.96     | 74.13    |
| Unet Resnet | 64          | 16         | 0.00013       | 0.5          |   | 64.51     | 63.13    |

**Notes:** We still have plan to attempt to submit our result as *late submission* in this competition, that final result will be announced here.

# [Tracking experiment](#experiment)

# [Usage](#usage)

# [Pretrained weight](#weight)

# [Team member](#member)
**Dung Manh Nguyen (me)**
- Github: [manhdung20112000](https://github.com/manhdung20112000)
- Email: [manhdung20112000@gmail.com](manhdung20112000@gmail.com)

**Giang Pham Truong**
- Github: [giangTPham](https://github.com/giangTPham)

**Tran Trung Thanh**
- Github: [amaggat](https://github.com/amaggat)

# [Reference](#refer)
<a id="1">[1]</a> Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional
Networks for Biomedical Image Segmentation. 2015. arXiv: 1505.04597 [cs.CV].

<a id="2">[2]</a> Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for
Large-Scale Image Recognition. 2015. arXiv: 1409.1556 [cs.CV].