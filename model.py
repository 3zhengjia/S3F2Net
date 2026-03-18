# 导入必要的库：PyTorch神经网络模块、函数式接口、数值计算库、PyTorch核心库
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# 设置计算设备：优先使用GPU（cuda:0），否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """
    封装标准卷积层（带padding，保证尺寸不变）
    Parameters:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        kernel_size (int): 卷积核大小，默认3
        stride (int): 步长，默认1
        padding (int): 填充大小，默认1（保持卷积后尺寸不变）
        dilation (int): 膨胀率，默认1
        groups (int): 分组卷积数，默认1（普通卷积）
    Returns:
        nn.Conv2d: 配置好的卷积层（无偏置，后续用BN替代偏置作用）
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class GCN_1(nn.Module):
    """图卷积网络分支1：实现自适应邻接矩阵的图卷积操作"""

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        # 输入维度为8，输出维度为8，A为关联矩阵
        super(GCN_1, self).__init__()
        self.A = A  # 初始邻接矩阵
        self.BN = nn.BatchNorm1d(input_dim)  # 一维BatchNorm：对节点特征做归一化
        self.Activition = nn.LeakyReLU()  # 激活函数：LeakyReLU避免梯度消失
        # 可学习参数：sigma1（控制邻接矩阵相似度缩放）
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 线性层1：将节点特征映射到256维，用于计算自适应邻接矩阵
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        # 线性层2：将节点特征映射到输出维度，用于图卷积输出
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]  # 节点数（超像素数量）
        # 单位矩阵I：保证每个节点与自身相连（自环）
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # 邻接矩阵掩码：保留原始邻接矩阵的连接结构（非零位置为1，零位置为0）
        self.mask = torch.ceil(self.A * 0.00001)  # 小系数避免浮点误差，ceil将非零值转为1

    def A_to_D_inv(self, A: torch.Tensor):
        """
        计算邻接矩阵的度矩阵的逆平方根（用于图卷积的归一化）
        Parameters:
            A (torch.Tensor): 邻接矩阵（shape[节点数, 节点数]）
        Returns:
            torch.Tensor: 度矩阵的逆平方根（shape[节点数, 节点数]）
        """
        D = A.sum(1)  # 计算度向量：每行求和（节点的度）
        D_hat = torch.diag(torch.pow(D, -0.5))  # 度矩阵的逆平方根（对角矩阵）
        return D_hat

    def forward(self, H, model='normal'):
        """
        GCN前向传播：自适应邻接矩阵 + 图卷积计算
        Parameters:
            H (torch.Tensor): 输入节点特征（shape[节点数, 输入维度]）
            model (str): 模式选择（'normal'为默认，其他模式做额外裁剪）
        Returns:
            torch.Tensor: GCN输出特征（shape[节点数, 输出维度]）
            torch.Tensor: 自适应邻接矩阵（shape[节点数, 节点数]）
        """
        # 方案一：minmax归一化的图卷积（带自适应邻接矩阵）
        H = self.BN(H)  # 节点特征归一化
        H_xx1 = self.GCN_liner_theta_1(H)  # 特征映射到256维
        # 计算自适应邻接矩阵：特征内积→sigmoid（相似度）→掩码保留连接结构→加自环
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        if model != 'normal':
            A = torch.clamp(A, 0.1)  # 特殊模式下裁剪邻接矩阵值（避免过小值）

        # 图卷积归一化：D^(-0.5) * A * D^(-0.5)（对称归一化）
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        # 图卷积操作：归一化邻接矩阵 × 特征映射后的节点特征
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)  # 激活函数

        # # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)  # 用于mask非连接边
        # mask=torch.ceil(A * 0.00001)
        # nodes_count =A.shape[0]
        # I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # A = torch.where(mask > 0, e, zero_vec) + I  # 只保留连接边的相似度
        # if model != 'normal': A = torch.clamp(A, 0.1)
        # A = F.softmax(A, dim=1)  # 行归一化（softmax）
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A


class GCN_2(nn.Module):
    """图卷积网络分支2：与GCN_1结构相同，作为串行GCN的第二级"""

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCN_2, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        if model != 'normal':
            A = torch.clamp(A, 0.1)

        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)

        return output, A


class S3F2Net(nn.Module):
    """S3F2Net主网络：融合CNN（局部特征）和GCN（全局特征）的双分支网络"""

    def __init__(self, Q, A, FM, NC, Classes):
        """
        初始化S3F2Net
        Parameters:
            Q (torch.Tensor): 像素-超像素关联矩阵（shape[总像素数, 超像素数]）
            A (torch.Tensor): 超像素邻接矩阵（shape[超像素数, 超像素数]）
            FM (int): 卷积特征图基础通道数
            NC (int): 输入图像通道数（如高光谱图像的波段数）
            Classes (int): 分类任务的类别数
        """
        super(S3F2Net, self).__init__()
        self.Q = Q  # 像素-超像素关联矩阵
        self.A = A  # 超像素邻接矩阵
        self.Classes = Classes  # 类别数
        input_dim = 8  # GCN输入特征维度
        output_dim = 8  # GCN输出特征维度
        dim = 8  # 去噪卷积的输出维度

        # ===================== CNN分支1：处理主输入x1（如高光谱图像） =====================
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=NC, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),  # 二维BatchNorm：对卷积特征图归一化
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 最大池化：尺寸减半，通道数不变
            # nn.Dropout(0.5),  # 可选dropout防止过拟合
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),  # 通道数翻倍
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),  # Dropout：随机失活50%神经元
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),  # 通道数再翻倍
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # ===================== CNN分支2：处理辅助输入x2（如空间特征图） =====================
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 1, 1),  # 单通道输入→FM通道
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),  # LeakyReLU激活
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        # ===================== 共享权重卷积层：融合两个CNN分支特征 =====================
        self.conv7 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        # ===================== 去噪卷积层：处理GCN输入x3（如原始像素特征） =====================
        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN', nn.BatchNorm2d(1))  # 输入归一化
        self.CNN_denoise.add_module('CNN_denoise_Conv',
                                    nn.Conv2d(1, dim, kernel_size=(3, 3), stride=1, padding=1))  # 去噪卷积
        self.CNN_denoise.add_module('CNN_denoise_Act', nn.LeakyReLU())  # 激活

        # ===================== 分类头：CNN分支的输出分类层 =====================
        self.out1 = nn.Linear(FM * 4, Classes)  # 分支1分类头
        self.out2 = nn.Linear(FM * 4, Classes)  # 分支2分类头
        self.out3 = nn.Linear(FM * 4, Classes)  # 融合分支分类头

        # ===================== GCN相关初始化 =====================
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 按列维度归一化
        self.linear_1 = nn.Linear(dim, 1)  # GCN特征评分层（节点重要性评分）
        self.linear_2 = nn.Linear(self.A.size(0), 1)  # 阈值计算层（自适应阈值）
        self.linear_3 = nn.Linear(dim, self.Classes)  # GCN分类头
        self.pooling = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化（全局特征聚合）
        self.GCN_Branch_1 = GCN_1(input_dim, output_dim, self.A)  # GCN分支1实例化
        self.GCN_Branch_2 = GCN_2(input_dim, output_dim, self.A)  # GCN分支2实例化

    def forward(self, x1, x2, x3):
        """
        前向传播：CNN局部特征提取 + GCN全局特征提取 + 局部-全局融合分类
        Parameters:
            x1 (torch.Tensor): 主输入（如高光谱图像，shape[B, NC, H, W]，B为批次）
            x2 (torch.Tensor): 辅助输入（如空间特征图，shape[B, 1, H, W]）
            x3 (torch.Tensor): GCN输入（如原始像素特征，shape[H, W, dim]）
        Returns:
            tuple: 三个融合分类结果（out1, out2, out3）
        """
        # ===================== CNN分支1：处理x1 =====================
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        xh = self.conv3(x1)  # 分支1最终特征（高维局部特征）

        # ===================== CNN分支2：处理x2 =====================
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        xl = self.conv6(x2)  # 分支2最终特征（低维局部特征）

        # ===================== 共享权重卷积融合 =====================
        x1_fusion = self.conv7(x1)  # 分支1特征经共享卷积
        x2_fusion = self.conv7(x2)  # 分支2特征经共享卷积
        x_fusion = x1_fusion + x2_fusion  # 特征相加融合

        # ===================== CNN局部特征分类 =====================
        xh = xh.view(xh.size(0), -1)  # 展平：[B, FM*4, H', W'] → [B, FM*4],经过三次池化后，样本变成1x1的了
        CNN_result_1 = self.out1(xh)  # 分支1分类结果

        xl = xl.view(xl.size(0), -1)  # 展平分支2特征
        CNN_result_2 = self.out2(xl)  # 分支2分类结果

        x_fusion = x_fusion.view(x_fusion.size(0), -1)  # 展平融合特征
        CNN_result_3 = self.out3(x_fusion)  # 融合分支分类结果

        # ===================== GCN分支：处理x3（全局特征提取） =====================
        # 1. 输入去噪：x3→[1, 1, H, W]→去噪卷积→[1, dim, H, W]→调整维度→[H, W, dim]
        denoised_input = self.CNN_denoise(torch.unsqueeze(x3.permute([2, 0, 1]), 0))  #（1，8，349，1905）
        denoised_input = torch.squeeze(denoised_input, 0).permute([1, 2, 0])  #（349，1905，8）

        # 2. 像素特征→超像素特征：展平像素特征→通过关联矩阵Q映射到超像素维度
        flattened_input = denoised_input.view(denoised_input.size(0) * denoised_input.size(1),
                                              denoised_input.size(2))  # [H*W, dim]
        initial_features = torch.mm(self.norm_col_Q.t(), flattened_input)  # [z,d]

        # 3. 第一级GCN：提取初始全局特征
        H1, _ = self.GCN_Branch_1(initial_features)
        scores = self.linear_1(H1)  # 节点重要性评分：[超像素数, 1]

        # 4. 自适应阈值筛选：排序评分→计算阈值→筛选重要节点特征      #写的有毛病，可以改
        sorted_scores, sorted_indices = torch.sort(scores, dim=0)  # 按评分升序排序
        threshold = self.linear_2(sorted_scores.permute([1, 0]))  # 计算自适应阈值
        sorted_scores = sorted_scores - threshold  # 评分减去阈值（突出重要节点）
        _, inverse_sorted_indices = torch.sort(sorted_indices, dim=0)  # 表示原始分数再升序排列里的位置
        weighted_features = sorted_scores * H1[sorted_indices].squeeze()  # 按评分加权特征
        recovered_features = weighted_features[inverse_sorted_indices].squeeze()  # 恢复原始节点顺序

        # 5. 余弦相似度矩阵：计算节点间特征相似度（增强全局关联性）
        similarity_matrix = F.cosine_similarity(H1.unsqueeze(1), H1.unsqueeze(0), dim=-1)  # [超像素数, 超像素数]

        # 6. 特征融合：相似度矩阵加权→残差连接（保留原始特征）
        Integrated_features = torch.matmul(similarity_matrix, recovered_features)
        updated_features = Integrated_features + recovered_features  # 残差增强

        # 7. 第二级GCN：细化全局特征
        H2, _ = self.GCN_Branch_2(updated_features)

        # 8. 超像素特征→像素特征：通过关联矩阵Q映射回像素维度
        GCN_result = torch.matmul(self.Q, H2)  # [总像素数, dim]
        GCN_result = GCN_result.permute([1, 0])  # [dim, 总像素数]

        # 9. 全局特征聚合与分类
        GCN_result = self.pooling(GCN_result)  # 自适应池化：[dim, 1]
        GCN_result = GCN_result.permute([1, 0])  # [1, dim]
        GCN_result = self.linear_3(GCN_result)  # GCN分类结果：[1, Classes]
        GCN_result = GCN_result.repeat(CNN_result_1.size(0), 1)  # 匹配批次维度：[B, Classes]
        GCN_result = F.softmax(GCN_result, dim=1)  # 归一化为概率分布

        # ===================== 局部-全局特征融合分类 =====================
        out1 = torch.mul(CNN_result_1, GCN_result)  # 分支1局部特征 × GCN全局特征
        out2 = torch.mul(CNN_result_2, GCN_result)  # 分支2局部特征 × GCN全局特征
        out3 = torch.mul(CNN_result_3, GCN_result)  # 融合局部特征 × GCN全局特征

        return out1, out2, out3