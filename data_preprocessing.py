import pandas as pd
import numpy as np

df = pd.read_csv('/Users/ramtahor/Desktop/projects/kaggel/dataset/uw-madison-gi-tract-image-segmentation/train.csv')
print(df)





def rleToMask(rleString,height,width):
  rows,cols = height,width
  rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
  rlePairs = np.array(rleNumbers).reshape(-1,2)
  img = np.zeros(rows*cols,dtype=np.uint8)
  for index, length in rlePairs:
    index -= 1
    img[index:index+length] = 255
  img = img.reshape(cols,rows)
  img = img.T
  return img

def create_mask_images(df: pd.DataFrame, img_dir):
    for i, row in df.iterrows():

'dataset/uw-madison-gi-tract-image-segmentation/train/case123/case123_day20/scans/slice_0001_266_266_1.50_1.50.png'