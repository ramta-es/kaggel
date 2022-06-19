import rasterio.plot
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


class MriDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        pass

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json_file_before', type=str, help='path to polygons csv file to parse')
    parser.add_argument('json_file_after', type=str, help='path to polygons csv file to parse')
    parser.add_argument('root', type=str, help='path to images root')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
    args = parser.parse_args()
    print(vars(args))

    dataset = HsDataset(root=args.root, csv_path=args.csv_file)
    x = dataset[2]
    # csv_file = pd.read_csv()
