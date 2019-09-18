import glob
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow
import keras
from keras import backend as K
import os
from pathlib import Path
from segmentation_models import Unet
train_dir_str = 'C:\\Users\\SATHGS\\PycharmProjects\\Steel_defect\\dataset\\train_images\\'
test_dir_str = 'C:\\Users\\SATHGS\\PycharmProjects\\Steel_defect\\dataset\\test_images\\'
train_dir = glob.glob(train_dir_str)
test_dir = glob.glob(test_dir_str)
seq = tensorflow.keras.utils.Sequence

train = pd.read_csv('train.csv')
train_size = len(
    glob.glob(train_dir[0] + '*.jpg'))
train_img_filenames = glob.glob(r'C:\Users\SATHGS\PycharmProjects\Steel_defect\dataset\train_images\*.jpg')
image_size = np.array(Image.open(train_img_filenames[0])).shape


def rle_decode(rle, image_size):
    rle_array = rle.split()
    start_points = [int(x)-1 for x in rle_array[0::2]]
    length = [int(x) for x in rle_array[1::2]]
    end_points = [sum(x) for x in zip(start_points, length)]

    mask = np.zeros(image_size[0]*image_size[1], dtype=int)
    for start_point,end_point in zip(start_points,end_points):
        mask[start_point:end_point] = 255
    img_size = (image_size[0], image_size[1])
    mask = np.reshape(mask, img_size, order='F')

    return mask


def combine_masks(rles):
    is_present = np.isnan(rles)
    mask = np.empty((256, 1600, 4), dtype=np.int8)
    for i, j in enumerate(is_present):
        if j == 'True':
            mask[:, :, i] = rle_decode(rles(i), (256, 1600))

    return mask


def modify_gt_table(df):
    df["Image_ID"] = df["ImageId_ClassId"].apply(lambda x: x.split('_')[0])
    new_df = pd.DataFrame({"Image_ID": df["Image_ID"][::4]})
    new_df["class_1"] = df["EncodedPixels"][::4].values
    new_df["class_2"] = df["EncodedPixels"][1::4].values
    new_df["class_3"] = df["EncodedPixels"][2::4].values
    new_df["class_4"] = df["EncodedPixels"][3::4].values
    new_df["Encoded_Pixels"] = new_df[['class_1', 'class_2', 'class_3', 'class_4']].values.tolist()
    new_df.reset_index(inplace=True, drop=True)
    new_df.fillna('', inplace=True)
    #new_df['count'] = np.sum(new_df.iloc[:, 1:] != '', axis=1).values

    return new_df


class data_generator(keras.utils.Sequence):
    def __init__(self, df, batch_size, mode='fit'):               #LABELS = [[]*4] from call
        self.image_filenames, self.labels = df["Image_ID"].values, df["Encoded_Pixels"].values
        self.df = df
        self.batch_size = batch_size
        self.shuffle = True
        self.mode = mode

    def __len__(self):
        return np.ceil(len(self.image_ids) / float(self.batch_size))

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, 256, 1600, 3), dtype=np.float32)
        Y = np.empty((self.batch_size, 256, 1600, 4), dtype=np.int8)
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        for i, index in enumerate(indexes):
            img = Image.open(train_dir + self.df["Image_ID"].iloc[index])
            X[i, ] = img
            if self.mode == 'fit':
                Y[i, ] = combine_masks(self.df["Encoded_Pixels"].iloc[index])

        if self.mode == 'fit':
            return X, Y
        else:
            return X


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)










