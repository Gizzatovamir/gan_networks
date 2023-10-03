import lightning
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np
import pathlib
from PIL import Image
import torch


class HorseDataSet(Dataset):
    def __init__(self, img_list: np.ndarray, image_dir: pathlib.Path, transform=None):
        self.transform = transform
        self.img_list = img_list
        self.img_dir = image_dir

    def __len__(self) -> np.ndarray:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img_path = self.img_dir.name + self.img_list[index]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(img_path).convert("RGB")
        image = image.resize((64, 64))
        label = label.resize((64, 64))
        image = np.asarray(image, dtype=np.float32) / 255
        label = np.asarray(label, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        return image, label
