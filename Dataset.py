import torch
import os
import time
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import logging
from log import setup_log, logger
from torch.utils.data import Dataset, DataLoader, Subset

from GI import preprocess_hadamard, CGI

class GI_Dataset(Dataset):
    def __init__(self, dataset, M, img_size, speckle_matrix, name):
        self.dataset = dataset
        self.max_samples = len(dataset)
        self.sample = M
        self.size = img_size
        self.speckle_matrix = speckle_matrix
        self.name = name
        
        # 预先计算所有样本的CGI重构结果
        self.precomputed_cgi = []
        self.precompute_cgi()
    
    def precompute_cgi(self):
        total_time = 0
        for i in range(self.max_samples):
            if i == 0:
                gi_start_time = time.time()
            # 获取原始图像
            image, label = self.dataset[i]
            # logger.debug(image.size())
            
            # 转换为numpy数组并调整形状，这里squeeze()会删除数值是1的维度，因此单通道图像的通道信息会被抹去
            img_np = image.squeeze().numpy()

            # 调用CGI重构函数，传入预设的采样次数和Hadamard矩阵
            gi_reconstructed = CGI(self.sample, self.speckle_matrix, img_np)
            
            # 转换回张量，并且对于单通道灰度图像，将原本的通道维度复原
            gi_tensor = torch.tensor(gi_reconstructed, dtype=torch.float32).unsqueeze(0)
            # gi_tensor = torch.tensor(gi_reconstructed, dtype = torch.float32)
            
            self.precomputed_cgi.append((gi_tensor, image))

            if (i+1) % 10000 == 0:
                logger.debug(f"Completed: {i+1}/{self.max_samples}")
                gi_end_time = time.time() 
                # logger.debug(f"1000 ghost images completed in {gi_end_time - gi_start_time:.4f} seconds.\n")
                total_time += gi_end_time - gi_start_time
                gi_start_time = time.time()
        logger.debug(f"{self.name} ghost images precomputed in {total_time:.4f} seconds.\n")
        torch.save(self.precomputed_cgi, 'precomputed_cgi.pt')

    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # 这里就是对应的元组，包含gi桶探测值和原始图像两个张量
        return self.precomputed_cgi[idx]


def get_dataloader(M, img_size, speckle_matrix, batch_size):
    train_imgs = datasets.STL10(
        root = '../dataset/SPI_SwinTransformer/STL-10/train_imgs', 
        split = 'unlabeled', 
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(), 
            transforms.Grayscale(num_output_channels = 1), 
            # transforms.Normalize((0.4497, ), (0.2262, )), 
            ])
        )

    # train_imgs = datasets.MNIST(
    #     root = '../dataset/SPI_SwinTransformer/MNIST/train_imgs', 
    #     # split = 'train', 
    #     train = True, 
    #     download = True, 
    #     transform = transforms.Compose([
    #         transforms.Resize((img_size, img_size)), 
    #         transforms.ToTensor(), 
    #         # transforms.Grayscale(num_output_channels = 1), 
    #         # transforms.Normalize((0.4497, ), (0.2262, )), 
    #         ])
    #     )


    # 随机选取80%作为训练集，20%作为验证集
    train_size = int(len(train_imgs) * 0.8)
    indices = list(range(len(train_imgs)))
    np.random.shuffle(indices)
    train_data = Subset(train_imgs, indices[:train_size])
    valid_data = Subset(train_imgs, indices[train_size:])

    train_dataloader = DataLoader(
        GI_Dataset(train_data, M, img_size, speckle_matrix, name = 'Train', ), 
        batch_size = batch_size, 
        shuffle = True, 
        )
    valid_dataloader = DataLoader(
        GI_Dataset(valid_data, M, img_size, speckle_matrix, name = 'Validate', ), 
        batch_size = batch_size, 
        shuffle = True, 
        )

    '''这里返回的是Pytorch的DataLoader对象，包含了各个批次
    每个批次的结构是一个包含两个张量的元组(batch_gi, batch_original_images)，前者是桶探测值，后者是原始图像
    对batch_gi，其类型是[B, C, M]；对batch_original_images，其类型是[B, C, H, W]'''
    return train_dataloader, valid_dataloader