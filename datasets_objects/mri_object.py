import pandas as pd
import numpy as np
from pathlib import Path
import imageio
import matplotlib.pyplot as plt


class Mri_Object(object):

    def __init__(self, df: pd.DataFrame, im_path: Path, sep: str):
        self.path = im_path
        self.id = self.get_id(self.path, sep)

        self.df = df.loc[df.index == self.id, ['segmentation', 'class']]

        self.df = self.df.set_index('class')
        self.label_dict = {'small_bowel': 1, 'large_bowel': 2, 'stomach': 3}

        self.image = np.array(imageio.imread(self.path)).astype(np.float64)
        self.image = self.image[np.newaxis, :, :]
        print('image shape', self.image.shape)

    def create_mask_images(self):

        mask = np.zeros((self.image.shape[1], self.image.shape[2]))

        for label, row in self.df.iterrows():
            if type(self.df.at[label, 'segmentation']) != float:
                mask2 = self.rleToMask((self.df.at[label, 'segmentation']), self.image.shape[1],
                                       self.image.shape[2], self.label_dict[label])
                print('mask', mask.shape)
                print('mask2', mask2.shape)
                mask = np.maximum(mask, mask2.T)

            else:

                continue

        return mask.T[np.newaxis, :, :]

    @staticmethod
    def rleToMask(rleString, height, width, val):
        rows, cols = height, width
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index:index + length] = val
        img = img.reshape(cols, rows)
        return img

    @staticmethod
    def get_id(file_name, sep: str):
        id_list = str(file_name).split('/')
        id = (id_list[4] + '_' + (id_list[6].split('.png')[0]).split('.')[0]).split('_')
        id = sep.join(id[0:4])

        return id  # file id as written in the dataframe: string
