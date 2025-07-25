import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Subset, DataLoader  
from torchvision import datasets, transforms  
from torchvision.transforms import ToTensor  
from skimage.metrics import structural_similarity as ssim
import torch.amp as amp

from SPI_SwinTransformer import SPISwinTransformer
from Dataset import get_dataloader

def train(dataloader1, dataloader2, model, loss_fn, optimizer, device, epoch, epochs, train_path1, train_path2, valid_path1, valid_path2):
    batches1 = len(dataloader1)
    batches2 = len(dataloader2)
    MAX = 1.0

        # 初始化 GradScaler
    scaler = amp.GradScaler() # 在 train 函数开始时初始化

    # 训练阶段
    model.train()
    train_loss_value = []
    train_loss = 0.
    train_psnr = 0.
    train_ssim = 0.

    for batch_idx, (x, y) in enumerate(dataloader1):
        '''这里x对应self.precomputed_cgi[idx][0]，y对应self.precomputed_cgi[idx][1]'''
        x, y = x.to(device), y.to(device)
        # x = x.squeeze(1)# .squeeze(-1) # 将 [batch_size, 1, M, 1] 变为 [batch_size, M]

        with amp.autocast('cuda'): # 自动混合精度
            pred = model(x)
            # 修正：直接使用y，因为它已经是灰度图
            loss = loss_fn(pred, y) # <-- 修正：这里使用y

        optimizer.zero_grad()
        scaler.scale(loss).backward() # <-- 使用scaler.scale(loss)
        scaler.step(optimizer)        # <-- 使用scaler.step(optimizer)
        scaler.update()               # <-- 更新scaler
        train_loss += loss.item()
        batch_psnr = 0.
        batch_ssim = 0. 

        for i in range(x.size(0)):
            # 对于 PSNR/SSIM 计算
            y_np = y[i].squeeze().cpu().detach().numpy()
            pred_np = pred[i].squeeze().cpu().detach().numpy()
            
            mse = np.mean((y_np - pred_np) ** 2)
            if mse < 1e-10:
                PSNR = 100
            else:
                PSNR = 10 * np.log10(MAX**2 / mse)
            batch_psnr += PSNR
            
            # SSIM 需要使用data_range参数确保图像范围一致
            SSIM = ssim(y_np, pred_np, data_range = MAX)
            batch_ssim += SSIM

            # 保存首个epoch和末个epoch中的首个batch的图像作为对比
            batch_dir1 = ""
            batch_dir2 = ""
            if epoch in (0, epochs - 1) and batch_idx in (0, batches1 - 1):
                batch_dir1 = os.path.join(train_path1, f"batch{batch_idx + 1}")
                batch_dir2 = os.path.join(train_path2, f"batch{batch_idx + 1}")
                os.makedirs(batch_dir1, exist_ok = True)
                os.makedirs(batch_dir2, exist_ok = True)
                Y = y_np
                Pred = pred_np
                if np.max(Y) - np.min(Y) > 1e-8:
                    Y  = 255 * (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
                if np.max(Pred) - np.min(Pred) > 1e-8:
                    Pred = 255 * (Pred - np.min(Pred)) / (np.max(Pred) - np.min(Pred))
                cv2.imwrite(os.path.join(batch_dir1, f"pic{i + 1}.png"), Y.astype(np.uint8))
                cv2.imwrite(os.path.join(batch_dir2, f"pic{i + 1}.png"), Pred.astype(np.uint8))

        train_psnr += batch_psnr / x.size(0)
        train_ssim += batch_ssim / x.size(0)

    train_avg_loss = train_loss / batches1
    train_loss_value.append(train_avg_loss)
    train_avg_psnr = train_psnr / batches1
    train_avg_ssim = train_ssim / batches1

    print(f'\nTraining Epoch {epoch+1}')
    print('-' * 30)
    print(f"MSE Loss: {train_avg_loss:.6f}")
    print(f"PSNR: {train_avg_psnr:.4f} dB, SSIM: {train_avg_ssim:.4f}")

    # 验证阶段
    model.eval()
    valid_loss_value = []
    valid_loss = 0.
    valid_psnr = 0.
    valid_ssim = 0.

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader2):
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)# .squeeze(-1) # 将 [batch_size, 1, M, 1] 变为 [batch_size, M]

            with amp.autocast('cuda'): # 自动混合精度
                pred = model(x)
                loss = loss_fn(pred, y) # <-- 修正

            valid_loss += loss.item()
            batch_psnr = 0.
            batch_ssim = 0. 
        
            for i in range(x.size(0)):
                y_np = y[i].squeeze().cpu().detach().numpy()
                pred_np = pred[i].squeeze().cpu().detach().numpy()
            
                mse = np.mean((y_np - pred_np) ** 2)
                if mse < 1e-10:
                    PSNR = 100
                else:
                    PSNR = 10 * np.log10(MAX**2 / mse)
                batch_psnr += PSNR
            
                # SSIM 需要使用data_range参数确保图像范围一致
                SSIM = ssim(y_np, pred_np, data_range = MAX)
                batch_ssim += SSIM
            
                batch_dir1 = ""
                batch_dir2 = ""
                # 保存首个epoch和末个epoch中的首个batch的图像作为对比
                if epoch in (0, epochs - 1) and batch_idx in (0, batches2 - 1):
                    batch_dir1 = os.path.join(valid_path1, f"batch{batch_idx + 1}")
                    batch_dir2 = os.path.join(valid_path2, f"batch{batch_idx + 1}")
                    os.makedirs(batch_dir1, exist_ok = True)
                    os.makedirs(batch_dir2, exist_ok = True)
                    Y = y_np
                    Pred = pred_np
                    if np.max(Y) - np.min(Y) > 1e-8:
                        Y  = 255 * (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
                    if np.max(Pred) - np.min(Pred) > 1e-8:
                        Pred = 255 * (Pred - np.min(Pred)) / (np.max(Pred) - np.min(Pred))
                    cv2.imwrite(os.path.join(batch_dir1, f"pic{i + 1}.png"), Y.astype(np.uint8))
                    cv2.imwrite(os.path.join(batch_dir2, f"pic{i + 1}.png"), Pred.astype(np.uint8))

            valid_psnr += batch_psnr / x.size(0)
            valid_ssim += batch_ssim / x.size(0)

        valid_avg_loss = valid_loss / batches2
        valid_loss_value.append(valid_avg_loss)
        valid_avg_psnr = valid_psnr / batches2
        valid_avg_ssim = valid_ssim / batches2

        print(f'\nValidating Epoch {epoch+1}')
        print('=' * 30)
        print(f"MSE Loss: {valid_avg_loss:.6f}")
        print(f"PSNR: {valid_avg_psnr:.4f} dB, SSIM: {valid_avg_ssim:.4f}")

        # plt.plot(train_loss_value, label = 'Training Loss')
        # plt.plot(valid_loss_value, label = 'Validation Loss')
        # plt.xlable('Epoch')
        # plt.legend(loc = 'best')
        # plt.show()