import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import os

class BeamCKM(Dataset):
    def __init__(self, maps_inds=np.zeros(1), phase="train", dir_dataset="datasetPath", 
                 numTx=100, simulation="data", transform=None):
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 100, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1, self.ind2 = 0, 79
        elif phase == "val":
            self.ind1, self.ind2 = 80, 89
        elif phase == "test":
            self.ind1, self.ind2 = 90, 99
        else:
            raise ValueError("phase必须是'train', 'val'或'test'")

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.simulation = simulation

        # 路径配置
        self.dir_buildings = os.path.join(dir_dataset, "png/buildings_complete/")
        self.dir_Tx = os.path.join(dir_dataset, "png/antennas/")
        self.dir_gain = os.path.join(dir_dataset, f"{simulation}/")

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):
        idxr = idx // self.numTx
        idxc = idx % self.numTx
        map_id = self.maps_inds[idxr + self.ind1] + 1

        input_buildings = io.imread(os.path.join(self.dir_buildings, f"{map_id}.png")) / 255.0
        input_Tx = io.imread(os.path.join(self.dir_Tx, f"{map_id}_{idxc}.png")) / 255.0
        inputs = np.stack([input_buildings, input_Tx], axis=2)  # shape: [256, 256, 2]
        inputs = np.transpose(inputs, (2, 0, 1))  # shape: [2, 256, 256]

        image_gain_multi = np.zeros((256, 256, 8))
        for beam_id in range(8):
            gain_path = os.path.join(self.dir_gain, f"{map_id}_{idxc}/{beam_id}.png")
            image_gain_multi[..., beam_id] = io.imread(gain_path) / 255.0
        image_gain_multi = np.transpose(image_gain_multi, (2, 0, 1))
        return inputs, image_gain_multi