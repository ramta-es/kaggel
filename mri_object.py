import pandas as pd
import numpy as np
from pathlib import Path
import imageio


def get_id(file_name, sep: str):
    id_list = str(file_name).split('/')
    id = (id_list[4] + '_' + (id_list[6].split('.png')[0]).split('.')[0]).split('_')
    id = sep.join(id[0:4])

    return id  # file id as written in the dataframe: string


class Mri_Object(object):

    def __init__(self, df: pd.DataFrame, im_path: Path, sep: str):
        self.path = im_path
        self.id = get_id(self.path, sep)

        self.df = df.loc[df.index == self.id, ['segmentation', 'class']]

        self.df = self.df.set_index('class')

        self.image = np.array(imageio.imread(self.path))

    def create_mask_images(self):
        label_dict = {'small_bowel': 1, 'large_bowel': 2, 'stomach': 3}
        mask = np.zeros((self.image.shape[0], self.image.shape[1]))

        for label, row in self.df.iterrows():
            print(label)
            if type(self.df.at[label, 'segmentation']) != float:
                mask = mask[self.rleToMask((self.df.at[label, 'segmentation']), self.image.shape[0],
                                           self.image.shape[1], label_dict[label]) != 0]
                print('mask', np.max(mask))
            else:
                continue

        return mask

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
        img = img.T
        return img
