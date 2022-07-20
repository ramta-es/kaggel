
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path

import imgaug
from collections import defaultdict
import imgaug.augmenters as iaa
import random
import torch
import hs.data_processing as hsu

df = pd.read_csv('/Users/ramtahor/Desktop/projects/kaggel/dataset/uw-madison-gi-tract-image-segmentation/train.csv')
img_root = 'dataset/uw-madison-gi-tract-image-segmentation/train'

class MriDataset(Dataset):
    def __init__(self, images_root, csv_path):
        self.seg_df = pd.read_csv(csv_path)
        self.images = images
        self.labels = self.seg_df['class']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image_tensor = 0
        label_tensor = 0
        return image_tensor, label_tensor

    def get_dataloaders(self, batch_size, train_split=0.8):
        pass

    def rleToMask(self, rleString, height, width):
        rows, cols = height, width
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index:index + length] = 255
        img = img.reshape(cols, rows)
        img = img.T
        return img

    def get_images_and_labels(self, img_path, csv_path):
        im_list = []
        label_list = []
        label = {'class': 0, 'number': 1, 'date': 2}
        for file in Path(img_root).glob('*/*.png'):
            image = np.load(str(file))[:, :, band]
            image = img(images=(image.reshape(image.shape[2], image.shape[0], image.shape[1])))
            im_list.append(image.astype(np.float32))
            label_list.append(int(((str(file).split('.npy')[0]).split('class-')[1]).split('_')[i]))
        return np.array(im_list), np.array(label_list)

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
dataset = MriDataset()
x = dataset[2]
# csv_file = pd.read_csv()
