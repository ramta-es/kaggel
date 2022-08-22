from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from datasets_objects.mri_object import Mri_Object
from typing import Union, Iterable
from collections import defaultdict
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

im_root = 'dataset/uw-madison-gi-tract-image-segmentation'
csv_path = '/dataset/uw-madison-gi-tract-image-segmentation/train.csv'
csv_path = '/dataset/uw-madison-gi-tract-image-segmentation/train.csv'


class MriDataset(Dataset):
    def __init__(self, im_root, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.set_index('id')
        self.images = []
        self.masks = []
        for file in Path(im_root).glob('*/*/*/*/*.png'):
            self.img = Mri_Object(df=self.df, im_path=file, sep='_')
            self.images.append(self.img.image)
            self.masks.append(self.img.create_mask_images())
            if len(self.images) > 10:
                break

        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images) * train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


# Uncomment for check
# dataset = MriDataset(im_root, csv_path)


class SpectralCube(Dataset):
    def __init__(self, images_path, method, band: Union[int, Iterable]):
        self.images, self.labels = self.get_image_and_label(images_path, band, method)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])

    @staticmethod
    def get_image_and_label(directory: str, band: Union[int, Iterable], method: str):
        im_list = []
        label_list = []
        label = {'class': 0, 'number': 1, 'date': 2}
        i = label[method]
        for file in Path(directory).glob('*/*.npy'):
            image = np.load(str(file))[:, :, band]
            img = iaa.Sequential([
                iaa.Resize({"height": 224, "width": 224}),  # resize image
                # iaa.AdditiveGaussianNoise(),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.ScaleX((0.5, 1.5)),  # rescale along X axis
                iaa.ScaleY((0.5, 1.5)),  # rescale along Y axis
                iaa.Rotate((-45, 45)),  # rotate randomly in -45 45 degrees
                iaa.Fliplr(0.5)]  # horizontally flip 50% of the images
                # iaa.GaussianBlur(sigma=(0, 3.0))])  # blur images with a sigma of 0 to 3.0
            )

            image = img(images=(np.transpose(image, (2, 0, 1))))

            im_list.append(image.astype(np.float32))
            label_list.append(int(((str(file).split('.npy')[0]).split('class-')[1]).split('_')[i]))
        return np.array(im_list), np.array(label_list)

    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images) * train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


















# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('json_file_before', type=str, help='path to polygons csv file to parse')
#     parser.add_argument('json_file_after', type=str, help='path to polygons csv file to parse')
#     parser.add_argument('root', type=str, help='path to images root')
#     parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
#     args = parser.parse_args()
#     print(vars(args))

# img = iaa.Sequential([
#     iaa.Resize({"height": 224, "width": 224}),  # resize image
#     # iaa.AdditiveGaussianNoise(),  # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.ScaleX((0.5, 1.5)),  # rescale along X axis
#     iaa.ScaleY((0.5, 1.5)),  # rescale along Y axis
#     iaa.Rotate((-45, 45)),  # rotate randomly in -45 45 degrees
#     iaa.Fliplr(0.5)]  # horizontally flip 50% of the images
#     # iaa.GaussianBlur(sigma=(0, 3.0))])  # blur images with a sigma of 0 to 3.0
# )

# dataset = MriDataset(root=args.root, csv_path=args.csv_file)

# x = dataset[2][1].shape
# print('shape', x)
