from os.path import isfile
from src import config
import matplotlib.pyplot as plt
from PIL import Image
import torch
import random
import os
import numpy as np


def expand_path(path):
    path = str(path)

    if isfile(path) :
        return path

    if isfile(config.TRAIN_PATH + path + ".png"):
        return config.TRAIN_PATH + (path + ".png")

    if isfile(config.PREV_TRAIN_PATH + path + '.jpeg'):
        return config.PREV_TRAIN_PATH + (path + ".jpeg")

    if isfile(config.TEST_PATH + path + ".png"):
        return config.TEST_PATH + (path + ".png")

    return path


def p_show(imgs, train_df, label_name=None, per_row=3):
    n = len(imgs)
    rows = (n + per_row - 1) #pery_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(15,15))

    for ax in axes.flatten(): 
        ax.axis('off')

    for i,(p, ax) in enumerate(zip(imgs, axes.flatten())): 
        img = Image.open(expand_path(p))
        ax.imshow(img)
        ax.set_title(train_df[train_df.id_code == p].diagnosis.values)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True