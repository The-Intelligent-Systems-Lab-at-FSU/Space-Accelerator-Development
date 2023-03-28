import os
from glob import glob

import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image

import torch


classes = {"15_orless_conf": 0,
           "100_conf": 1}


class Fire(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.__data = path
        self.__files = glob(os.path.join(path, "**", "*.tif"), recursive=True)

        self.__labels = [classes[str(f.lstrip(path).split("/")[0])]
                         for f in self.__files if "Aqua" not in f]

        self.__transforms = transforms

    def __getitem__(self, index):
        label = self.__labels[index]
        img = rio.open(self.__files[index])

        if self.__transforms is not None:
            img = self.__transforms(img)

        return img, label

    def __len__(self):
        return len(self.__files)
