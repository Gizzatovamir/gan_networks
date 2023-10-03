import pytorch_lightning
from Dataloader import Dataset
import os
import yaml

DATASET_PATH = "./horse_dataset"

if __name__ == "__main__":
    dir_list = os.listdir(DATASET_PATH)
    for data_set_dir in dir_list:
        pass
