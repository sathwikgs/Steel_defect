import glob
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# train_dir = 'C:\Users\SATHGS\PycharmProjects\Steel_defect\dataset\train_images'
# test_dir = 'C:\Users\SATHGS\PycharmProjects\Steel_defect\dataset\test_images'

train = pd.read_csv('train.csv')
train_size = len(
    glob.glob(r'C:\Users\SATHGS\PycharmProjects\Steel_defect\dataset\train_images\*.jpg'))
train_img_filenames = glob.glob(r'C:\Users\SATHGS\PycharmProjects\Steel_defect\dataset\train_images\*.jpg')
image_size = np.array(Image.open(train_img_filenames[0])).shape

def rle_decode(rle, image_size):
    rle_array = rle.split()
    print(rle_array)
    start_points = [int(x)-1 for x in rle_array[0::2]]
    length = [int(x) for x in rle_array[1::2]]
    end_points = [sum(x) for x in zip(start_points, length)]

    mask = np.zeros(image_size[0]*image_size[1], dtype=int)
    for start_point,end_point in zip(start_points,end_points):
        mask[start_point:end_point] = 255
    img_size = (image_size[0], image_size[1])
    mask = np.reshape(mask, img_size, order='F')

    return mask

mask_image = rle_decode(train.iloc[0,1], image_size)
print (mask_image.shape)
mask_image = Image.fromarray(mask_image)
mask_image.show()












