import glob
import numpy as np
import pandas as pd

train_dir = '/dataset/train_images/'
test_dir = '/dataset/test_images/'

train = pd.read_csv('train.csv')
train_size = len(
    glob.glob(train_dir + "*.jpg"))
train_img_filenames = glob.glob(train_dir + "*.jpg")

no_defect_images = 0
defect_images = 0

class_count = {1: 0, 2: 0, 3: 0, 4: 0}

for i in range(0, len(train.index), 4):

    labels = train.iloc[i:i+4, 1]
    if (labels.isna().all()):
        no_defect_images += 1
    else:
        defect_images += 1

    for index, label in enumerate(labels.isna().values.tolist()):
        if (label == False):
            class_count[index + 1] += 1







