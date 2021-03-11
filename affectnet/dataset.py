# this file is taken from https://github.com/yutarochan/emotic/blob/master/src/util/data.py
import os
import sys
import time
import torch
import warnings
import numpy as np
from PIL import Image
import scipy.io as sio
from PIL import ImageFile
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as tF
import torch.utils.data as data
import pandas as pd
import json

emotion_dic = {
    "angry": 0,
    "disgust": 1,
    "disgusted": 1,
    "fear":  2,
    "fearful": 2,
    "happy": 3,
    "sad":   4,
    "surprise": 5,
    "surprised": 5,
    "neutral": 6
}


class AffectNet(data.Dataset):
    def __init__(self, root, csv_path, transform=None):
        self.transform = transform

        self.df = pd.read_csv(csv_path)
        self.root = root
        self.transform = transform


        self.df = self.df[self.df.expression!=8]
        self.df = self.df[self.df.expression!=9]
        self.df = self.df[self.df.expression!=10]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        im = Image.open(os.path.join(self.root, sample['subDirectory_filePath'])).convert('RGB')

        left = sample['face_x']
        top = sample['face_y']
        width = sample['face_width']
        height = sample['face_height']


        face = tF.crop(im, top, left, height, width)


        if self.transform is not None:
            im = self.transform(im)
            face = self.transform(face)

  
        return face, sample['expression']

