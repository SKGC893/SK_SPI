import numpy as np
import torch
from scipy.linalg import hadamard
import torch.amp as amp

# GPU设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} to reconstruct gi now.\n")

def preprocess_hadamard(size):
    """生成Hadamard矩阵并转换为PyTorch张量"""
    if not (size > 0 and (size & (size - 1)) == 0):
        raise ValueError("Size must be a power of 2.")
    
    hadamard_original = torch.from_numpy(hadamard(size**2)).float().to(device)
    binary_matrix0 = (hadamard_original + 1) // 2
    binary_matrix1 = 1 - binary_matrix0
    
    return hadamard_original, binary_matrix0, binary_matrix1

@torch.no_grad()
def CGI(M, hadamard_original, img_np):

    # # 假设 img_np 的维度顺序是 (通道数, 高度, 宽度) 或 (批次大小, 通道数, 高度, 宽度)
    # if img_np.ndim == 3 and img_np.shape[0] == 3: # 单张RGB图像 (C, H, W)
    #     # 使用标准的亮度转换公式将RGB转换为灰度图：Y = 0.2989*R + 0.5870*G + 0.1140*B
    #     # 先将维度转置为 (H, W, C) 以便进行矩阵乘法
    #     img_np_grayscale = np.dot(img_np.transpose((1, 2, 0)), [0.2989, 0.5870, 0.1140])
    #     # 添加一个通道维度，使其变为 (1, H, W)
    #     img_np = np.expand_dims(img_np_grayscale, axis=0)
    # elif img_np.ndim == 4 and img_np.shape[1] == 3: # 批次RGB图像 (B, C, H, W)
    #     # 处理批次图像的灰度化
    #     img_np_grayscale_batch = np.dot(img_np.transpose((0, 2, 3, 1)), [0.2989, 0.5870, 0.1140])
    #     # 添加一个通道维度，使其变为 (B, 1, H, W)
    #     img_np = np.expand_dims(img_np_grayscale_batch, axis=1)


    # 将输入图像转换为GPU张量
    img_tensor = torch.from_numpy(img_np).float().to(device)
    
    # 检查输入有效性
    if M > hadamard_original.shape[0]:
        raise ValueError("N must be <= hadamard_original rows count.")
    
    # 选择前M个Hadamard模式
    # 问题：我试图将一个1048576大小的数据重构为[1024, 3, 32, 32]
    # 这里是通道数的问题，他那边弄出来是1通道的但是由于STL-10本来是RGB图像所以是3通道
    S = hadamard_original[:M].view(M, *img_np.shape)  # [M, H, W]
    
    # 散斑调制和桶探测值计算 (全部在GPU上完成)
    with amp.autocast('cuda'):
        modulate = S * img_tensor  # 广播乘法 [M, H, W]
        bucket_values = modulate.sum(dim=(-1, -2))  # 沿空间维度求和 [M]
    
    # 返回CPU端的numpy数组
    return bucket_values.cpu().numpy()      # 这个数据类型会是[M, 1]，就是DL4GI中的M个一维桶探测值的序列