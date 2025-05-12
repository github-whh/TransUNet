import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import os

class RadioUNet_c(Dataset):
    """适配RadioWnet的DataLoader，支持8通道输出"""
    def __init__(self, maps_inds=np.zeros(1), phase="train", dir_dataset="/home/haohan/Mywork/Code/RadioMapSeer/", 
                 numTx=100, simulation="data", transform=None):
        """
        Args:
            phase: "train", "val", "test"
            dir_dataset: 数据集路径
            numTx: 每个地图的发射器数量
            transform: 数据增强
        """
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 100, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        # 划分训练/验证/测试集
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
        # 计算地图和发射器索引
        idxr = idx // self.numTx
        idxc = idx % self.numTx
        map_id = self.maps_inds[idxr + self.ind1] + 1

        # 加载输入数据（基站位置图+环境图）
        input_buildings = io.imread(os.path.join(self.dir_buildings, f"{map_id}.png")) / 255.0
        input_Tx = io.imread(os.path.join(self.dir_Tx, f"{map_id}_{idxc}.png")) / 255.0
        inputs = np.stack([input_buildings, input_Tx], axis=2)  # 形状: [256, 256, 2]
        inputs = np.transpose(inputs, (2, 0, 1))  # 形状: [2, 256, 256]

        # 加载8通道的标签数据（假设已预处理为8个波束的RadioMap）
        # 注意：需确保数据集中存在对应的多通道标签文件（如`gain_beam0.png`, `gain_beam1.png`, ...）
        image_gain_multi = np.zeros((256, 256, 8))
        for beam_id in range(8):
            gain_path = os.path.join(self.dir_gain, f"{map_id}_{idxc}/{beam_id}.png")
            image_gain_multi[..., beam_id] = io.imread(gain_path) / 255.0  # 归一化
        image_gain_multi = np.transpose(image_gain_multi, (2, 0, 1))
        return inputs, image_gain_multi