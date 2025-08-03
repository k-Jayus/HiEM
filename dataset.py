import os
import pickle
import nibabel as nib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, subject, data_csv, train=True):
        self.train = train
        self.data_csv = data_csv
        self.mri_path = [root + i for i in os.listdir(root)]

        subject = torch.load(subject)
        ad = subject["ad"]
        label_ad = [1] * len(ad)
        cn = subject["mci"]
        label_cn = [0] * len(cn)

        X = ad + cn
        Y = label_ad + label_cn
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=True, random_state=43
        )

        if self.train:
            self.sample = x_train
            self.label = y_train
        else:
            self.sample = x_test
            self.label = y_test

    def __getitem__(self, item):
        sample = self.sample[item]
        label = self.label[item]
        for path in self.mri_path:
            if sample in path:
                mri = sitk.ReadImage(path)
                arr = sitk.GetArrayFromImage(mri)
                if (arr.max() - arr.min()) ==0:
                    print(sample)
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                label = torch.tensor(label, dtype=torch.int64)

                return arr, label,sample

    def __len__(self):
        if self.train:
            return len(self.sample)
        else:
            return len(self.sample)


if __name__ == "__main__":
    root = "./ADNI/"
    subject = "./files/subject.pt"
    data_csv = "./files/data.csv"

    train_set = MyDataset(root, subject, data_csv, train=False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
    for (mri, data), label in loader:
        print("mri: shape={} min={} max={}".format(mri.shape, mri.min(), mri.max()))
        print("data: shape={}".format(data.shape))
        print("label={} shape={}".format(label, label.shape))
        print("------------------")
