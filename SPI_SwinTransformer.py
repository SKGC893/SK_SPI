from re import S
from tkinter import SE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


# 在Swin Transformer之前的CNN预处理，生成特征图并提取低级特征
class ShallowFeature(nn.Module):
    def __init__(self, sampling_times, img_size, dropout):
        super().__init__()
        self.sampling_times = sampling_times
        self.img_size = img_size

        self.fc = nn.Sequential(
            nn.Linear(self.sampling_times, self.img_size ** 2), 
            nn.Dropout(dropout), 
            )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = 3, padding = 'same'), 
            nn.BatchNorm2d(1), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            )
    
    def forward(self, x):
        # 层级结构简单，残差结构不必要
        x = self.fc(x)
        x = x.view(-1, 1, self.img_size, self.img_size)
        output = self.conv(x)
        return output


# 深层特征提取，轻量化卷积可以不使用丢弃法
class DeepFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 'same'), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),  
            )

    def forward(self, x):
        output = self.conv(x)
        return output


# 重构回灰度GI图像
class LastBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size = 3, padding = 'same'), 
            nn.BatchNorm2d(1), 
            nn.ReLU(), 
            )

    def forward(self, x):
        x = self.upsample(x)
        output = self.conv(x)
        return output


class DropPath(nn.Module):
    '''DropPath，也称为 Stochastic Depth，即随机深度
    随机深度的核心思想其实和dropout是类似的，都是在自己所处的层面进行随机丢弃，从而提升模型性能
    dropout是丢弃某个层内部的神经元，droppath则是丢弃某些层/路径/残差块
    因为在Transformer中，往往会有很多堆叠的注意力和MLP构成残差块，整体机制更为复杂
    因此DropPath更像是dropout的变体，更针对这种具有复杂残差结构的深度网络'''
    def __init__(self, drop_prob = 0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob      # 丢弃的概率

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob  # 计算保留的概率
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 构建生成随机mask的shape，并且保证能够处理不同维度的张量，而不仅仅是二维卷积神经网络

        '''随机张量的值在[keep_prob, 1 + keep_prob)之间，之后向下取整
        这样小于1.0的值（原本torch.rand中小于drop_prob）的部分会变成0
        大于等于1.0的值会变成1，得到了一个由0和1组成的二值mask
        也就是说有drop_prob这么多部分是0，之后应用到矩阵上就相当于“丢弃”'''
        random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
        random_tensor.floor()       # random_tensor最后成为一个二值的mask，为0的概率是drop_prob，为1的概率是keep_prob

        '''这里的操作涉及到一个尺度补偿的操作，dropout中其实也有
        比如原本输出期望是x，丢弃p过后，期望就变成(1-p)*x + p*0 = (1-p)*x
        也就是说每次丢弃之后信号强度都会衰减p，最终会导致信号消失
        因此对保留的部分，将其除以(1-p)，保证期望始终是x'''
        output = x.div(keep_prob) * random_tensor
        return output

def window_partition(x, window_size):
    '''将特征图分割成窗口，不能重叠也不能有空隙'''
    B, H, W, C = x.shape
    '''一个H*W的大特征图，可以看作是由H // window_size行和W // window_size列的大块组成的
    每个大块的大小就是window_size * window_size
    重塑后，将张量重构成了[B, H_num_windows, window_size, W_num_windows, window_size, C]的形式
    相当于是把整个特征图看作了H_num_windows*W_num_windows的窗口网格，每个窗口是window_size*window_size'''
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # permute操作是为了将窗口的维度调整为[B * num_windows, window_size, window_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    '''将窗口合并成特征图，不能重复也不能有空隙
    输入windows的形状是[num_windows * B, window_size, window_size, C]'''
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    '''这里就是ViT继承过来的重要部分了 
    为了让transformer能够处理图像而不是NLP中的序列，ViT将图像分成许多patches
    之后，每个patch都当作一个token。这个模块的作用就是将图像转换为token
    
    参数说明：
        - patch_size: 每个图像块的变长
        - in_c: 输入通道数
        - embed_dim: 输出token的特征维度'''
    def __init__(self, patch_size = 4, in_c = 1, embed_dim = 96, norm_layer = None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_c
        self.embed_dim = embed_dim

        '''使用类似CNN的方式来实现图像块的分割
        根据patch_size设置卷积核大小和步长，确保采样不重叠也没有缝隙
        并且由于是卷积层，同时完成了分块和线性变换，之后归一化'''
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding，这里这么写是为了提高泛化能力，确保patch长宽不一致的情况也能处理
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:       # 如果H和W不是patch_size的整数倍，就进行填充
            # 对[B, C, H, W]的张量，填充顺序是W、H、C，B通常不填充
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 
                          0, self.patch_size[0] - H % self.patch_size[0], 
                          0, 0))

        # 下采样patch_size倍，这里是学习CNN的下采样过程
        x = self.proj(x)    # 这里的C已经变成embed_dim了
        _, _, H, W = x.shape
        # 将特征图展平，变成[B, C, H * W]的形式，然后转换成[B, HW, C]
        x = x.flatten(2).transpose(1, 2)    # 现在H*W就是patch数量，embed_dim是每个patch的特征维度，符合transformer输入格式
        output = self.norm(x)
        return output, H, W


class PatchMerging(nn.Module):
    '''PatchMerging是swin transformer中独有的模块
    主要是将多个patch合并成一个更大的patch，从而实现下采样
    这个模块的核心思想是将多个小patch合并成一个大patch，从而减少特征图的分辨率
    此外，由于是对特征图中每个不重叠的2*2区域进行操作的，所以H, W必须是2的整数倍
    
    这里patch分割融合的步骤是：
    1.先分割成小的patches，进行窗口注意力，捕捉精细的局部特征和细节
    2.合并成大的patches，再次进行窗口注意力，学习更泛化的全局特征
    swin transformer的“先分小再合并”的本质是计算量和准确性的权衡
    这也是区别于ViT的关键之一，他是仿照CNN的设计引入了分层结构的
    这样即能降低特征图分辨率，减少计算量，又能增加通道数，捕捉多尺度信息
    类比CNN的画，可以理解为一个特殊的池化层，不是简单的取平均或最大
    是将一定范围邻域内的特征沿着特征通道进行拼接，之后通过线性层进行维度压缩'''
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super().__init__()

        '''这里的dim参数容易混淆
        dim代表的是每个patch的特征维度（全称是特征向量的维度），可以理解为一个patch处理后表示为一个长度是dim的向量
        而这里的dim是对原本输入形状重构得到的
        比如说，最开始的输入是[B, C, H, W]，这里的C就是通道数，比如灰度图像C = 1
        经过patchembed之后，转换成了[B, L, C']
        这里的L = H*W / (patch_size*patch_size)，C'则是转换过后的每个patch的特征维度
        而在patchmerge中，需要将[B, L, C']重新转换为[B, H', W', C']，这里C'自然就成了中间表示的通道数
        '''
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias = False)

    def forward(self, x, H, W):
        # 将[B, L, C]转换成[B, H, W, C]的形式
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W %2, 0, H % 2))     # 在W左侧和H顶部进行填充

        '''PatchMerging的核心逻辑
        1.假设输入特征图 (H=4, W=4, C=96)：
        ┌───┬───┬───┬───┐
        │a1 │a2 │a3 │a4 │  每个字母代表一个patch
        ├───┼───┼───┼───┤  (例如a1=[96维向量])
        │b1 │b2 │b3 │b4 │
        ├───┼───┼───┼───┤
        │c1 │c2 │c3 │c4 │
        ├───┼───┼───┼───┤
        │d1 │d2 │d3 │d4 │
        └───┴───┴───┴───┘
        2.重新划分为2*2的不重叠的大patch
        ┌───────┬───────┐
        │ a1 a2 │ a3 a4 │
        │ b1 b2 │ b3 b4 │
        ├───────┼───────┤
        │ c1 c2 │ c3 c4 │
        │ d1 d2 │ d3 d4 │
        └───────┴───────┘
        3.每个2*2大patch内的4个小patch进行通道拼接
        左上块: concat(a1,b1,c1,d1) → [96*4=384维]
        右上块: concat(a2,b2,c2,d2) → [384维]
        左下块: concat(a3,b3,c3,d3) → [384维]
        右下块: concat(a4,b4,c4,d4) → [384维]
        4.使用线性层将384维压缩到192维，输出从4个小patch变成了1个大patch
        5.此外，新的patch相较于原本的patch，从96维变成了192维，所以实际上是通道数翻倍的
        6.至于总信息量，实际上确实是信息量减少了一半，但是这个减少是“精炼”之后的，相当于量减少了一半，但是信息密度翻倍了'''
        x0 = x[:, 0::2, 0::2, :]        # 左上角
        x1 = x[:, 1::2, 0::2, :]        # 左下角
        x2 = x[:, 0::2, 1::2, :]        # 右上角
        x3 = x[:, 1::2, 1::2, :]        # 右下角
        x = torch.cat([x0, x1, x2, x3], -1)     # 沿着最后一个维度（通道维度）拼接
        x = x.view(B, -1, 4 * C)        # [B, H_padded/2 * W_padded/2, 4 * C]

        x = self.norm(x)
        output = self.reduction(x)
        return output


class MLP(nn.Module):
    '''之前不是对特征图进行了线性变换吗
    MLP可以对每个token进行独立的非线性变换'''
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super(MLP, self).__init__()
        # 这里写or是为了提升灵活性
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        output = self.drop(self.fc2(x))
        return output


# 用这个窗口注意力模块替换原本的多头注意力
class WindowAttention(nn.Module):
    '''参数说明：
        - dim   输入通道数量
        - window_size   窗口的高度和宽度，是元组数据类型
        - num_heads 注意力头的个数
        - qkv_bias  是否要给qkv添加可学习的偏置
        - attn_drop 注意力权重的丢弃（失活）比率
        - proj_drop 输出的随机丢弃（失活）比率
        - head_dim  公式中的dk，每个注意力头的维度
        - scale 公式中的1/√dk，自注意力中常用的缩放因子，防止dk过大导致softmax梯度消失
    计算公式：Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k))V
    其中QK^T计算query和key之间的相似度，表示这个token对其他token的关注度
    sqrt(d_k)是缩放因子，防止点积过大导致softmax梯度过小
    之后乘以V得到加权求和后的value，表示当前token'''
    def __init__(self, dim, window_size, num_heads, qkv_bias = True, attn_drop = 0., proj_drop = 0., ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads     # 公式中的dk
        self.scale = head_dim ** -0.5       # 公式中的1/√dk

        '''SwinTransformer引入的可学习的相对位置偏置
        传统自注意力中，每个token都是等同的，模型无法区分位置信息
        使用相对位置偏置后，模型可以学习到不同位置之间的关系（比如token1在token2的右上角）'''
        
        '''相对位置偏置表的原理是，对一个长为M的一维序列，两个元素的相对距离有2M - 1种可能
        对于二维窗口，水平方向和垂直方向都分别有2Mw - 1和2Mh - 1种，一共有两者乘积种可能
        此外，还要对每个注意力头单独计算偏置表
        
        举例来说，假设window_size = 4, 那么2*4-1 = 7, 因此可能的相对位置范围是Δx, Δy∈[-3, 3]，组合起来共49种（7*7）
        偏移之后的范围是[0, 4)，令num_heads = 3，那么一共有49种相对位置，每个相对位置都有3个头的偏置值
        初始化就是下面这样，因为有3个头，所以每个位置都有3个值
        [ 
            [0.0, 0.0, 0.0],    index = 0(Δx = -3, Δy = -3)
            [0.0, 0.0, 0.0],    index = 1(Δx = -3, Δy = -2)
            [0.0, 0.0, 0.0],    index = 2(Δx = -3, Δy = -1)
            ...
            [0.0, 0.0, 0.0],    index = 48(Δx = 3, Δy = 3)
        ]
        之后会用截断正态分布初始化，然后开始学习这些偏置值
        '''
        self.relative_position_bias_table = nn.Parameter(       # 可学习的参数列表
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 尽可能中立的初始化相对位置偏置表，防止随机到某个极端的值
        nn.init.trunc_normal_(self.relative_position_bias_table, std = .02)

        '''计算相对位置索引
        window_size = 4, num_heads = 3举例，使用生成的高跟宽的张量构建坐标
        [(0,0), (0,1), (0,2), (0,3)
         (1,0), (1,1), (1,2), (1,3)
         (2,0), (2,1), (2,2), (2,3)
         (3,0), (3,1), (3,2), (3,3)]
         每个坐标的位置都对应一个patch
         
        但是上面是把行列坐标组合后的情况，实际上对于coords，有：
        coords[0] (行坐标):
        [[0,0,0,0],
         [1,1,1,1],
         [2,2,2,2],
         [3,3,3,3]]

        coords[1] (列坐标):
        [[0,1,2,3],
         [0,1,2,3],
         [0,1,2,3],
         [0,1,2,3]]
         
        展平之后是[2, 16]：
        [
         [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],  # 所有位置的行坐标
         [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]   # 所有位置的列坐标
        ]'''
        coords_h = torch.arange(self.window_size[0])        # 相对位置索引的高
        coords_w = torch.arange(self.window_size[1])        # 相对位置索引的宽
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing = 'ij'))     # 生成二维网格和对应坐标，[2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)       # 将二维网格展平为一维向量，方便后续计算相对位置索引，[2, Mh * Mw]

        '''广播相减得到所有patch的相对坐标
        由于广播机制，coords_flatten[:, :, None]是[2, 16, 1]
        coords_flatten[:, None, :]是[2, 1, 16]
        使用广播机制相减，得到所有位置对的坐标差[2, 16, 16]，第一维是两个坐标轴，第二维是16个query位置，第三维是16个key位置，注意这里每个“位置”都是一个位置坐标
        比如对(0, 0)位置(Q)，计算(0, 1)的相对坐标(K)就是(0, 0) - (0, 1) = (0, -1)
        再比如对(1, 2)位置(Q)，计算(3, 1)的相对坐标(K)就是(1, 2) - (3, 1) = (-2, 1)
        以此类推，每个坐标都要计算16次，都是用查询位置减去值的位置'''
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()     # 移位操作，确保直接选取前两维就能直接获得所有相对位置坐标
        
        '''偏移，确保所有坐标非负
        比如window_size = 4, 2*4-1 = 7，原本范围是[-3, 3]，偏移后变成[0, 6],横纵坐标都+3即可
        比如对计算后的(-2, 1)，偏移后是(1, 4)'''
        relative_coords[:, :, 0] += self.window_size[0] - 1     # 行方向偏移
        relative_coords[:, :, 1] += self.window_size[1] - 1     # 列方向偏移

        '''给每个独立的相对位置坐标都分配一个唯一的整数ID，确保能通过这个ID直接查找到对应的相对位置坐标
        计算公式是row*(2M-1)+col'''
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1     # 将所有行位置都乘以这个值
        relative_position_index = relative_coords.sum(-1)

        # 注册缓冲区，初始化的时候就会计算好，但是不会随着模型训练在反向传播中更新
        self.register_buffer("relative_position_index", relative_position_index)

        '''这里我再使用window_size = 2, num_heads = 3的情况做一个完整的图解
        1.生成坐标网格
        行坐标 (coords[0]):
        [[0,0],
         [1,1]]

        列坐标 (coords[1]):
        [[0,1],
         [0,1]]
        2.展平坐标
        行坐标: [0,0,1,1]
        列坐标: [0,1,0,1]
        3.计算相对坐标差
        Δx计算（行差）：
        Q\K (0,0) (0,1) (1,0) (1,1)
        (0,0)   0     0    -1    -1
        (0,1)   0     0    -1    -1
        (1,0)   1     1     0     0
        (1,1)   1     1     0     0
        Δy计算（列差）：
        Q\K (0,0) (0,1) (1,0) (1,1)
        (0,0)   0    -1     0    -1
        (0,1)   1     0     1     0
        (1,0)   0    -1     0    -1
        (1,1)   1     0     1     0
        4.坐标偏移
        Δx：
        [
         [1,1,0,0],
         [1,1,0,0],
         [2,2,1,1],
         [2,2,1,1]
        ]
        Δy：
        [
         [1,0,1,0],
         [2,1,2,1],
         [1,0,1,0],
         [2,1,2,1]
        ]
        5.转换为唯一索引（index = Δx * (2M - 1) + Δy），这里M = 2：
        [
         [4, 3, 1, 0],  # (0,0)对其他位置
         [7, 4, 2, 1],  # (0,1)对其他位置
         [7, 4, 2, 1],  # (1,0)对其他位置
         [8, 5, 3, 2]   # (1,1)对其他位置
        ]'''

        # 从这里开始就都一样了
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)     # 使用线性层将x投影成qkv三部分，dim * 3也是因为要生成3个向量
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 注意力分数归一化
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, mask = None):
        # x: [num_windows * batch_size, window_size * window_size, C]
        # x是从window_partition中得到并且展平了window_size * window_size的token序列
        B_, N, C = x.shape      # B_是总的窗口批次，N是每个窗口的token数量，C是特征维度
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)     # 提取qkv，形状都是[B_, num_heads, N, head_dim]
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = qkv.unbind(0)     # 和上面那个等效，但是兼容性更好
        attn = (q @ k.transpose(-2, -1)) * self.scale       # 根据公式计算

        '''这里第一个view将整个偏置表索引展平为一维长列表，其中包含了所有patch之间相对位置对应的唯一ID
        之后使用展平的偏置表索引去可学习的偏置表中查找对应偏置，每个注意力头一个
        第二个view则是将查找到的偏置值重塑回和注意力分数空间维度相匹配的张量形式，确保偏置能够正常放在注意力分数上
        之后将这个偏置张量直接加在计算出的注意力分数上，这样就能在模型中引入相对位置信息了'''
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果有mask，应用mask，是swin transformer移位窗口机制的关键
        if mask is not None:
            # SwinTransformer的写法
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        '''注意这里和swin transformer也不一样
        这里实际上就是更常见的transformer序列掩码形式'''
            # nW = mask.shape[0]
            # attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)       # 将mask应用在注意力分数上
            # attn = attn.view(-1, self.num_heads, N, N)      # 恢复原始数据形状
            # attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(0), float('-inf'))      # 将mask为true的地方填充为负无穷，保证softmax后是0
        # attn = attn.softmax(dim = -1)       # dim = -1 是key维度
        # attn = self.self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        output = self.proj_drop(x)
        return output


class SwinTransformerBlock(nn.Module):
    '''原本的TransformerLayer是简化后的SwinTransformer
    这里是完整版的SwinTransformer模块'''
    def __init__(self, dim, num_heads, window_size = 7, shift_size = 0, 
                 mlp_ratio = 4, qkv_bias = True, drop = 0., attn_drop = 0., drop_path = 0., 
                 act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size = (self.window_size, self.window_size), 
                                    num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x 
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        pad_l = pad_t = 0       # 只会在右侧和底部进行填充
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape      # 获取填充后的高度

        '''cyclic shift循环移位部分
        根据shift_size的大小判断是SW-MSA还是W-MSA的层
        shift_size == 0 说明窗口没有移位，就是W-MSA层'''
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_x = x
            attn_mask = None

        '''partition windows窗口分割部分
        将移位之后的特征图划分为互不重叠没有缝隙的窗口，计算窗口注意力'''
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask = attn_mask)

        '''merge windows窗口合并部分
        将计算注意力之后的各个窗口重新拼接为完整的特征图形式'''
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        '''reverse cyclic shift逆向循环移位
        之前不是将特征图进行了循环移位方便不同窗口之间计算注意力吗
        现在已经计算过注意力了，要将特征图恢复到初始的状态，因此逆运算'''
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            x = shifted_x
        
        # 移除padding填充的数据，恢复到原始的特征图大小
        if pad_r or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # 将恢复后的特征图展平，方便后续处理
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        output = x + self.drop_path(self.mlp(self.norm2(x)))
        return output


class SwinTransformerLayer(nn.Module):
    '''SwinTransformerLayer基础层
    参数说明：
        - dim: 输入通道数
        - depth: SwinTransformerBlock的数量
        - num_heads: 注意力头的数量
        - window_size: 窗口的大小
        - mlp_ratio: MLP的隐藏层维度与输入维度的比率
        - qkv_bias: 是否给qkv添加可学习的偏置
        - drop: dropout比率
        - attn_drop: 注意力权重的丢弃比率
        - drop_path: 随机深度的丢弃比率
        - norm_layer: 归一化层
        - downsample: 下采样模块
        - use_checkpoint: 是否使用检查点技术，减少内存消耗'''
    def __init__(self, dim, depth, num_heads, window_size, 
                 mlp_ratio = 4, qkv_bias = True, drop = 0., attn_drop = 0., 
                 drop_path = 0., norm_layer = nn.LayerNorm, 
                 downsample = None, use_checkpoint = False, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2      # 设计好的超参数，设置成window_size的一半既能保证计算效率又能提升模型性能

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim, 
                num_heads = num_heads, 
                window_size = window_size, 
                shift_size = 0 if(i % 2 == 0) else self.shift_size, 
                mlp_ratio = mlp_ratio, 
                qkv_bias = qkv_bias, 
                drop = drop, 
                attn_drop = attn_drop, 
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer = norm_layer)
            for i in range(depth)])

        # 在四个swin transformer block之后的卷积处理
        self.conv = DeepFeature(in_channels = dim, out_channels = dim)

        if downsample is not None:
            self.downsample = downsample(dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        '''创建滑动窗口自注意力机制的掩码'''
        # 首先保证Hp和Wp是window_size的整数倍，确保能应用于多尺度
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device = x.device)       # 确保排列顺序和特征图一致

        '''移位操作
        这里理解不畅通的根本问题是，比如layer1中划分了4个window，那么
        在layer2中，移位了两个patch，实际上就没有再分成4个window了
        但是如果这样的话，首先有8个window不是原本的大小需要重新填充，其次要计算更多的window计算量又上去了
        因此这里提出了新的计算方法：
        比如说你原来先划分了4个window
        a a a a | b b b b
        a a a a | b b b b
        a a a a | b b b b
        a a a a | b b b b
        --------+--------
        c c c c | d d d d
        c c c c | d d d d
        c c c c | d d d d
        c c c c | d d d d
        移位并重新进行编号（不用管内容，上面的0123就是编号）：
        ┌─────────────┬───────────────────┬─────────────┐
        │  0 (2×2)    │   1 (2×4)         │  2 (2×2)    │
        │  0  1       │   4  5  6  7      │ 12 13       │
        │  2  3       │   8  9 10 11      │ 14 15       │
        ├─────────────┼───────────────────┼─────────────┤
        │  3 (4×2)    │   4 (4×4)         │  5 (4×2)    │
        │ 16 17       │ 24 25 26 27       │ 40 41       │
        │ 18 19       │ 28 29 30 31       │ 42 43       │
        │ 20 21       │ 32 33 34 35       │ 44 45       │
        │ 22 23       │ 36 37 38 39       │ 46 47       │
        ├─────────────┼───────────────────┼─────────────┤
        │  6 (2×2)    │   7 (2×4)         │  8 (2×2)    │
        │ 48 49       │ 52 53 54 55       │ 60 61       │
        │ 50 51       │ 56 57 58 59       │ 62 63       │
        └─────────────┴───────────────────┴─────────────┘
        之后我们令区域0是A，区域3和6是B，区域1和2是C：
        ┌─────────────┬───────────────────┬─────────────┐
        │  A(0)       │   C(1)            │  C(2)       │
        │             │                   │             │
        │  0  1       │   4  5  6  7      │ 12 13       │
        │  2  3       │   8  9 10 11      │ 14 15       │
        ├─────────────┼───────────────────┼─────────────┤
        │  B(3)       │   4               │  5          │
        │             │                   │             │
        │ 16 17       │ 24 25 26 27       │ 40 41       │
        │ 18 19       │ 28 29 30 31       │ 42 43       │
        │ 20 21       │ 32 33 34 35       │ 44 45       │
        │ 22 23       │ 36 37 38 39       │ 46 47       │
        ├─────────────┼───────────────────┼─────────────┤
        │  B(6)       │   7               │  8          │
        │             │                   │             │
        │ 48 49       │ 52 53 54 55       │ 60 61       │
        │ 50 51       │ 56 57 58 59       │ 62 63       │
        └─────────────┴───────────────────┴─────────────┘
        先把A和C移动到下面，再把A和B移动到右边（这里太难画了，你去看pic1吧）
        移动之后，区域4是一个window，5和3是一个window，7和1是一个window，8620是一个window，最终还是4个window
        
        之后就可以使用掩码了，因为比如5和3构成的window，因为他俩实际上是分开的不是相邻的区域，直接进行注意力计算会出问题
        因此我们希望我们能在这个window中单独计算5和3的注意力，具体往下看'''
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        # 这里就是标记各个区域的ID
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        '''
        这里涉及到一个pytorch的广播机制，简单来说就是numpy会自动让维度不同无法加减的矩阵补齐让他们能够加减
        比如这里，我们令前面一个为A，后面一个为B
        A的维度是[nW, 1, Mh * Mw], B的维度是[nW, Mh * Mw, 1]
        那么这俩后两个维度不一样，没法直接减，就会应用广播机制补齐之后再相减
        比如A的后两维度相当于1行Mh * Mw列，那么就会把这一行复制Mh * Mw次，补齐成一个Mh * Mw行Mh * Mw列的矩阵
        对B也是同理，但是是对后两维的“列”的操作，补齐之后再相减
        最终结果是，shift前是相同窗口的patch的id会变为0，不是相同窗口的patch的id会变为非零的数，之后再进行后续处理就可以了
        
        这里的相减操作会触发广播机制，最终得到的是形状为[num_windows, N, N]的矩阵
        如果attn_mask中某个位置的值为0，说明token移位后仍在同一个窗口内（因为相同的减相同的才是0），可以计算注意力
        反之，如果不为0，说明移位之后不在同一个窗口内，不能计算注意力，要掩码掩盖掉
        注意这里的区域是通过ID来划分的，前面的img_mask已经标记好了
        
        续上上面的讲解，我们的目的是让区域5中的patch在和原本就是区域5中的patch计算注意力的时候更“注意”
        在原本区域5中的patch计算和原本区域3中的patch的注意力时更“不注意”
        这里看pic2，比如我们希望patch0在计算和patch0145891213的注意力时候更注意，而在计算和patch236710111415的注意力时候更不注意
        因此在计算出各个patch的注意力之后，将“不注意”的区域全部-100，让他变成很小的数
        这样在softmax之后，我们不需要的这些部分就会变成0，实际上我们还是只是计算了区域5中的注意力
        再比如对于区域4，由于内部数据本来就是连续的，因此可以直接计算，不仅没问题还实现了不同窗口的信息交互'''
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        '''把mask放在这里解决多尺度问题
        原本的SwinTransformer针对的是固定的输入尺寸
        如果输入尺寸和最初设定不同是无法正常生成mask的
        此外，这里和原图所给的stage不同，这里的写法是当前的stage和下一个stage的patch merging'''
        x0 = x
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
        # for (i, blk) in enumerate(self.blocks):
            blk.H, blk.W = H, W     # 动态传入了SwinTransformerBlock中的self.H和self.W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # 这里从blocks里出来是[B, L, C]，L = H*W
        B, L, C = x.shape
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)     # [B, C, H, W]
        x = self.conv(x)

        # 再重构回[B, L, C]的形式进行残差连接
        x = x.view(B, C, -1).permute(0, 2, 1)

        output = x0 + x
        if self.downsample is not None:
            # print("downsample")
            output = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return output, H, W


class SPISwinTransformer(nn.Module):
    def __init__(self, sampling_times, img_size, dropout,   # 这里是卷积预处理的参数
                 patch_size = 4, in_channels = 1, 
                 # num_classes = 1000,    # 这个num_classes是用在图像分类任务中的类别数
                 embed_dim = 96, 
                 depths = (2, 2, 6, 2),     # 每个stage中分别有几个block
                 num_heads = (3, 6, 12, 24),    # 每个stage中的注意力头数
                 window_size = 7, mlp_ratio = 4., qkv_bias = True, 
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1, 
                 norm_layer = nn.LayerNorm, patch_norm = True, 
                 use_checkpoint = False, **kwargs):
        super().__init__()
        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim ** 2 ** (self.num_layers - 1))
        self.num_features = int(embed_dim * (2 ** (self.num_layers - 1)))       # 因为上面那个占用内存过于离谱（28PB）所以改了
        self.mlp_ratio = mlp_ratio

        # 卷积预处理
        self.feature_map = ShallowFeature(sampling_times = sampling_times, img_size = img_size, dropout = dropout)
        self.patch_embed = PatchEmbed(
            patch_size = patch_size, in_c = in_channels, embed_dim = embed_dim, 
            norm_layer = norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p = drop_rate)

        '''设置随机深度衰减规则，即每层对应的丢弃比率
        表明DropPath的丢弃概率会随着网络加深而逐渐增大，是常见的正则化策略'''
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 这里好像和命名不太一样，那篇论文中是block包括layer，这里是layer包括block
            layers = SwinTransformerLayer(dim = int(embed_dim * 2 ** i_layer), 
                                          depth = depths[i_layer], 
                                          num_heads = num_heads[i_layer], 
                                          window_size = window_size, 
                                          mlp_ratio = self.mlp_ratio, 
                                          qkv_bias = qkv_bias, 
                                          drop = drop_rate, 
                                          attn_drop = attn_drop_rate, 
                                          drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                                          norm_layer = norm_layer, 
                                          downsample = PatchMerging if (i_layer < self.num_layers - 1) else None, 
                                          use_checkpoint = use_checkpoint)
            self.layers.append(layers)
        self.norm = norm_layer(self.num_features)
        self.conv1 = DeepFeature(in_channels = self.num_features, out_channels = self.num_features)
        self.conv2 = LastBlock(in_channels = self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 权重初始化，使用截断正态分布初始化线性层权重，偏重初始化为0
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.feature_map(x)
        x0 = x
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)

        # gemini给我的建议
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()   # [B, C, H, W]

        # 后面都正常
        x = self.conv1(x)
        x = x0 + x

        # 如果按照gemini的修改方式，这里就不需要再变形了
        # B, L, C = x.shape
        # x = x.view(B, L % 2, L % 2, C)    # [B, H, W, C]
        # x = x.permute(0, 3, 1, 2)   # [B, C, H, W]

        output = self.conv2(x)
        return output