import torch
import time
import os
import torch.optim as optim
from torch import nn

from GI import preprocess_hadamard
from SPItransformer import SPI_Transformer
from Dataset import get_dataloader
from Train import train

def main():
    '''
    超参数设置：
    M: 采样次数
    batch_size: 批次大小
    epochs: 训练轮数
    learning_rate: 学习率
    train_path1: 训练集原始图像存储路径
    train_path2: 训练集重构图像存储路径
    valid_path1: 验证集原始图像存储路径
    valid_path2: 验证集重构图像存储路径
    drop: dropout丢弃率
    '''
    M = 20      # 采样率2%
    batch_size = 16
    epochs = 40
    img_size = 32
    learning_rate = 0.0001
    drop = 0.1
    train_path1 = '../pictures/SPItransformer/train_images/original'
    train_path2 = '../pictures/SPItransformer/train_images/reconstruction'
    valid_path1 = '../pictures/SPItransformer/valid_images/original'
    valid_path2 = '../pictures/SPItransformer/valid_images/reconstruction'
    weights_path = 'weights'
    os.makedirs(train_path1, exist_ok = True)
    os.makedirs(train_path2, exist_ok = True)
    os.makedirs(valid_path1, exist_ok = True)
    os.makedirs(valid_path2, exist_ok = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} now.")
    print("In images preprocessing.\n")
    model = SPI_Transformer(M, img_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    speckle, _, _ = preprocess_hadamard(img_size)
    train_dataloader, valid_dataloader = get_dataloader(M, img_size, speckle, batch_size)

    # 尝试使用ReduceLROnPlateau动态调整学习率
    # learning_rate = 0.001
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode = 'min', 
    #     factor = 0.1, 
    #     patience = 3, 
    #     verbose = True, 
    #     min_lr = 1e-10, 
    #     threshold = 0.001, # threshold默认是按百分率计算的
    #     threshold_mode = 'rel'
    #     )

    total_time = 0
    # 模型训练
    for t in range(epochs):
        epoch_start_time = time.time()
        if t == 0:  print("\nTraining has been started.")
        train_path1_epoch = os.path.join(train_path1, f"epoch{t + 1}")
        train_path2_epoch = os.path.join(train_path2, f"epoch{t + 1}")
        valid_path1_epoch = os.path.join(valid_path1, f"epoch{t + 1}")
        valid_path2_epoch = os.path.join(valid_path2, f"epoch{t + 1}")
        train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, device, t, epochs, 
              train_path1_epoch, train_path2_epoch, valid_path1_epoch, valid_path2_epoch)
        epoch_end_time = time.time()
        print(f"Epoch {t + 1} completed in {epoch_end_time - epoch_start_time:.4f} seconds.\n")
        total_time += epoch_end_time - epoch_start_time
    print(f"Total training time: {total_time:.4f} seconds.\n")
    print("Training has been completed.\n")

    os.makedirs(weights_path, exist_ok=True)
    torch.save(model.state_dict(), f'weights/mnist_model_weights_sample={M}.pth')

if __name__ == '__main__':
    main()