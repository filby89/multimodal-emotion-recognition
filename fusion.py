import os
import numpy as np
import pandas as pd
import torch
from EmoReact.dataset import TSNDataSet
import torchvision


# example of results to fuse
results = [
    'logs//EmoReact/log/test RGB/0310_034133/',
    'logs//EmoReact/log/test Flow/0310_135737/']


label_names = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

f = TSNDataSet("test")
target = f.df[label_names]

continuous = f.df['Valence'].values/7.0

from model import metric
import sklearn.metrics

PATH = "/gpu-data/filby/EmoReact_V_1.0/experiments_tensorboard/"


cats = []
cats_val = []
pp = []
targets= []
for result in results:
    """in output_0.npy is always the output on the test set which runs at the end"""
    cat = np.load(os.path.join(PATH,result+"/output_{}.npy").format(0))

    cats.append(cat)
    roc_auc = metric.roc_auc(cat, target, average='macro')
    print("%s: ROC: %f" % (result, np.mean(roc_auc)))

print("Ensembles")

cat = np.mean(cats, axis=0)

if "val" in results[0] or True:
    roc_auc = metric.roc_auc(cat, target, average='micro')
    print("ROC: %f" % (np.mean(roc_auc)))
    print(metric.roc_auc(cat, target, average=None))
