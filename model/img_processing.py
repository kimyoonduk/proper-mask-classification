import pandas as pd
import os
import numpy as np
import joblib
from tqdm import tqdm

import torch
import random
import albumentations
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image
from tqdm import tqdm
from torchvision import models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# reference: https://debuggercafe.com/creating-efficient-image-data-loaders-in-pytorch-for-deep-learning/


def seed_everything(SEED=1337):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# get mean / std of each RGB channel for normalization
# no need to use anymore, since the computed values have been saved inside ImageDataset class
def norm_mean_std(X):
    images = []

    img_means = []
    img_stds = []

    resize = albumentations.Compose(
        [albumentations.Resize(img_size, img_size, always_apply=True)]
    )

    for i, img_path in tqdm(enumerate(X), total=len(X)):
        image = Image.open(img_path)
        image = resize(np.array(image))

        images.append(image)

    img_np = np.asarray(images)

    print(img_np.shape)
    # (10000, 256, 256, 3)

    for i in range(3):
        img_means.append(np.mean(img_np[:, :, :, i]))
        img_stds.append(np.std(img_np[:, :, :, i]))

    print(img_means)
    # [130.81978119506837, 114.49455226593018, 106.20569288635254]
    print(img_stds)
    # [70.6324415626536, 68.82429639461063, 73.3045452600786]

    # divide means and stds by 255
    img_means = np.asarray(img_means) / 255
    # [0.51301875, 0.44899824, 0.41649291]

    img_stds = np.asarray(img_stds) / 255
    # [0.27698997, 0.2698992, 0.2874688]

    return img_means, img_stds


# image dataset module
class ImageDataset(Dataset):

    # precomputed values for image normalization
    img_means = [0.51301875, 0.44899824, 0.41649291]
    img_stds = [0.27698997, 0.2698992, 0.2874688]

    def __init__(self, path, labels, img_size, train=True):
        self.X = path
        self.y = labels
        # apply augmentations
        if train:
            self.aug = albumentations.Compose(
                [
                    albumentations.Resize(img_size, img_size, always_apply=True),
                    albumentations.HorizontalFlip(p=1.0),
                    albumentations.ShiftScaleRotate(
                        shift_limit=0.3, scale_limit=0.3, rotate_limit=30, p=1.0
                    ),
                    albumentations.Normalize(
                        mean=img_means, std=img_stds, always_apply=True
                    ),
                ]
            )
        else:
            self.aug = albumentations.Compose(
                [
                    albumentations.Resize(img_size, img_size, always_apply=True),
                    albumentations.Normalize(
                        mean=img_means, std=img_stds, always_apply=True
                    ),
                ]
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )


def main():

    # set variables
    SEED = 1337
    seed_everything(SEED=SEED)
    target_ct = 2000
    img_size = 256
    data_path = "../data/input/data.csv"

    # whether to collapse 'incorrect' labels into a single class
    collapse_incorrect = False

    # set computation device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")

    # read the data.csv file and get the image paths and labels
    df = pd.read_csv(data_path)

    if collapse_incorrect:
        # all 'incorrect' labels are collapsed to '2' and have their sample size reduced to target_ct
        print("collapsing all 'incorrect' datapoints")
        df.at[df["target"] >= 2, "target"] = 2
        df1 = df[df["target"] < 2]
        df2 = df[df["target"] == 2]
        df2s = df2[:target_ct]
        df = pd.concat([df1, df2s])
        df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_path.values
    y = df.target.values

    (xtrain, xtest, ytrain, ytest) = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    train_data = ImageDataset(xtrain, ytrain, img_size=256, train=True)
    test_data = ImageDataset(xtest, ytest, img_size=256, train=False)

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False)

    print("Loading label binarizer...")
    lb = joblib.load("../data/input/lb.pkl")
