import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF
import torchvision.transforms as transforms



import librosa
import torchaudio
import matplotlib.pyplot as plt


class AudioDataSet(data.Dataset):
    def __init__(self, mode, transform=None):

        self.transform = transform

        # self.categorical_emotions = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

        # self.continuous_emotions = ["Valence"]
        
        self.df = pd.read_csv(os.path.join("EmoReact/ravdess_{}.csv".format(mode)))

        self.mode = mode

        self.embeddings = np.load("glove_embeddings.npy")


    def __getitem__(self, index):

        sample = self.df.iloc[index]
        # print(sample['path'])
        spec = np.load(sample['path'])

        spec = torch.FloatTensor(spec)
        # print(spec.shape)
        if self.transform:
            spec = self.transform(spec)

        categorical = int(self.df.iloc[index]['emotion'])-1


        return spec, torch.tensor(self.embeddings[:8]).float(), torch.tensor(categorical).float(), 0, 0, 0


    def __len__(self):
        return len(self.df)
