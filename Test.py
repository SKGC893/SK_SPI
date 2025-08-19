import torch
import os
import cv2
import numpy as np
import time
import logging
from log import setup_log, logger
from torch import nn
from torchvision import datasets, transforms  
from torchvision.transforms import ToTensor  
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader, Subset

from GI import preprocess_hadamard, CGI
from SPI_SwinTransformer import SPISwinTransformer
from Dataset import GI_Dataset

def test_loader(M, img_size, speckle_matrix, batch_size):
    # 这里之后考虑使用自定义数据集，比如Set12之类的
    test_imgs = datasets.STL10(
        root = '../dataset/SPI_SwinTransformer/STL-10/test_imgs', 
        split = 'test', 
        download = True, 
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(), 
            # transforms.Normalize((0.430, 0.417, 0.363), (0.231, 0.226, 0.223)), 
            transforms.Grayscale(num_output_channels = 1), 
            ])
        )

    # test_imgs = datasets.MNIST(
    #     root = '../dataset/SPI_SwinTransformer/MNIST/test_imgs', 
    #     # split = 'test', 
    #     train = False, 
    #     download = True, 
    #     transform = transforms.Compose([
    #         transforms.Resize((img_size, img_size)), 
    #         transforms.ToTensor(), 
    #         # transforms.Normalize((0.430, 0.417, 0.363), (0.231, 0.226, 0.223)), 
    #         transforms.Grayscale(num_output_channels = 1), 
    #         ])
    #     )

    test_data = test_imgs

    test_dataloader = DataLoader(
        GI_Dataset(test_data, M, img_size, speckle_matrix), 
        batch_size = batch_size, 
        shuffle = True, 
        )

    return test_dataloader

def test(dataloader, model, loss_fn, device, test_path1, test_path2):
    batches = len(dataloader)
    MAX = 1.0 

    model.eval()
    test_loss = 0.
    test_psnr = 0.
    test_ssim = 0.
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            batches = len(dataloader)
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # 计算损失
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            batch_psnr = 0.
            batch_ssim = 0.

            # 计算每个样本的 PSNR 和 SSIM
            for i in range(x.size(0)):
                y_np = y[i].squeeze().cpu().detach().numpy()
                pred_np = pred[i].squeeze().cpu().detach().numpy()

                mse = np.mean((y_np - pred_np) ** 2)
                if mse < 1e-10: 
                    PSNR = 100
                else:
                    PSNR = 10 * np.log10(MAX**2 / mse)
                batch_psnr += PSNR

                SSIM = ssim(y_np, pred_np, data_range=MAX) 
                batch_ssim += SSIM

                batch_dir1 = ""
                batch_dir2 = ""
                if batch_idx in (0, batches - 1):
                    batch_dir1 = os.path.join(test_path1, f"batch{batch_idx + 1}")
                    batch_dir2 = os.path.join(test_path2, f"batch{batch_idx + 1}")
                    os.makedirs(batch_dir1, exist_ok = True)
                    os.makedirs(batch_dir2, exist_ok = True)
                    Y = y_np
                    Pred = pred_np
                    if np.max(Y) - np.min(Y) > 1e-8:
                        Y = 255 * (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
                    if np.max(Pred) - np.min(Pred) > 1e-8:
                        Pred = 255 * (Pred - np.min(Pred)) / (np.max(Pred) - np.min(Pred))
                    cv2.imwrite(os.path.join(batch_dir1, f"pic{i + 1}.png"), Y.astype(np.uint8))
                    cv2.imwrite(os.path.join(batch_dir2, f"pic{i + 1}.png"), Pred.astype(np.uint8))
            
            test_psnr += batch_psnr / x.size(0) # 累加当前批次的平均PSNR
            test_ssim += batch_ssim / x.size(0) # 累加当前批次的平均SSIM

    # 计算平均损失、PSNR 和 SSIM
    test_avg_loss = test_loss / batches
    test_avg_psnr = test_psnr / batches
    test_avg_ssim = test_ssim / batches

    # 打印测试结果
    logger.debug("Final Test Results: ")
    logger.debug(f"MSE Loss: {test_avg_loss:.6f}")
    logger.debug(f"PSNR: {test_avg_psnr:.4f} dB")
    logger.debug(f"SSIM: {test_avg_ssim:.4f}")
    

def main():
    start_time = time.time()
    M = 1024
    batch_size = 16
    img_size = 64
    drop = 0.1      # 测试过程会关闭dropout
    speckle, _, _ = preprocess_hadamard(img_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"Using {device} now.")
    logger.debug("In images preprocessing.\n")

    model = SPISwinTransformer(M, img_size)
    pred_dict = torch.load(f'weights/stl10_model_weights_sample={M}.pth')
    # pred_dict = torch.load('weights/mnist_model_weights_sample=1024.pth')
    model.load_state_dict(pred_dict)
    model = model.to(device)
    loss_fn = nn.MSELoss()

    test_path1 = '../pictures/SPI_SwinTransformer/test_images/original'
    test_path2 = '../pictures/SPI_SwinTransformer/test_images/reconstruction'
    os.makedirs(test_path1, exist_ok = True)
    os.makedirs(test_path2, exist_ok = True)

    test_dataloader = test_loader(M, img_size, speckle, batch_size)

    test(test_dataloader, model, loss_fn, device, test_path1, test_path2)

    end_time = time.time()
    logger.debug(f"Total testing time: {end_time - start_time:.4f} seconds.\n")

if __name__ == '__main__':
    main()