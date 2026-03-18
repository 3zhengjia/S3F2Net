# 导入必要的库：系统操作、数值计算、随机数、矩阵保存、PyTorch核心、数据加载、神经网络、绘图、数据读取、PCA降维、时间、自定义模型/数据/工具函数、超像素分割
import os
import numpy as np
import random
import numpy.random
from scipy.io import savemat
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
import time
from model import S3F2Net  # 自定义S3F2Net模型
from data_pre import data_load, nor_pca, border_inter, con_data, getIndex, con_data1  # 数据预处理函数
from utils import output_metric  # 评价指标计算函数
import superpixel_seg  # 超像素分割模块

# 设置计算设备：优先使用GPU（cuda:0），否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== 实验参数设置 =====================
# 随机种子：保证实验结果可复现
seed = 0
torch.manual_seed(seed)  # 设置CPU随机种子
torch.cuda.manual_seed(seed)  # 设置GPU随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
random.seed(seed)  # 设置Python随机种子
np.random.seed(seed)  # 设置NumPy随机种子
os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子

# 训练参数
batchsize = 64  # 批次大小
EPOCH = 250  # 训练轮数
LR = 0.001  # 学习率
FM = 32  # CNN特征图基础通道数
seg_scale = 600  # 超像素分割的期望超像素数量
dataset_name = "Houston"  # 数据集名称（Houston数据集）

# ===================== 数据加载与预处理 =====================
# 加载数据集：返回高光谱数据(Data)、LiDAR数据(Data2)、训练标签(TrLabel)、测试标签(TsLabel)
Data, Data2, TrLabel, TsLabel = data_load(name=dataset_name)
# 获取LiDAR数据的尺寸（高度、宽度）
img_row = len(Data2)
img_col = len(Data2[0])

# 数据归一化与PCA降维：将数据映射到[0,1]并做PCA降维
[m, n, l] = Data.shape  # m:高光谱图像高度, n:宽度, l:波段数
PC, Data2, NC = nor_pca(Data, Data2, ispca=True)  # PC:PCA降维后的高光谱数据, data2:激光雷达数据 NC:PCA降维后的通道数

# ===================== LiDAR数据超像素分割 =====================
LiDAR_seg = Data2  # 基于LiDAR数据进行超像素分割
LiDAR_seg = torch.tensor(LiDAR_seg).float()  # 转换为张量
LiDAR_seg = np.expand_dims(LiDAR_seg, axis=-1)  # 扩展维度：[H,W]→[H,W,1]（添加通道维度）
ls = superpixel_seg.Superpixel_Seg(seg_scale)  # 初始化超像素分割器（期望seg_scale个超像素）
Q, S, A, Segments = ls.SGB(LiDAR_seg)  # 执行SGB超像素分割：Q(像素-超像素关联矩阵)、S(超像素均值特征)、A(超像素邻接矩阵)、Segments(超像素标签)
# 将矩阵移到GPU（用于模型计算）
Q = torch.from_numpy(Q).to(device)
A = torch.from_numpy(A).to(device)

# ===================== 边界插值与训练/测试集构建 =====================
# 边界插值：处理图像边界像素，避免边缘效应
x, x2 = border_inter(PC, Data2, NC)
# 构建高光谱(HSI)和LiDAR的训练/测试样本（按标签裁剪像素块）
TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2 = con_data(x, x2, TrLabel, TsLabel, NC)
print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)  # 打印HSI训练/测试集尺寸
print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)  # 打印LiDAR训练/测试集尺寸

# ===================== 数据格式转换（NumPy→PyTorch张量） =====================
# 高光谱训练数据转换：float32张量，标签减1（类别从0开始）并转为long型
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel) - 1
TrainLabel1 = TrainLabel1.long()

# 高光谱测试数据转换
TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel) - 1
TestLabel1 = TestLabel1.long()
Classes = len(np.unique(TrainLabel))  # 获取类别数

# LiDAR训练数据转换
TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2) - 1
TrainLabel2 = TrainLabel2.long()

# 创建训练数据集加载器：整合HSI、LiDAR训练数据和标签，支持批次加载与打乱
dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

# LiDAR测试数据转换
TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2) - 1
TestLabel2 = TestLabel2.long()

# ===================== 模型初始化与配置 =====================
# 创建S3F2Net模型实例：传入关联矩阵Q、邻接矩阵A、特征图通道数FM、输入通道数NC、类别数Classes
model = S3F2Net(Q, A, FM=FM, NC=NC, Classes=Classes)
model.cuda()  # 将模型移到GPU加速训练

# 初始化优化器与损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam优化器（自适应学习率）
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失（适用于分类任务）

# ===================== 训练与测试循环 =====================
BestAcc = 0  # 记录最佳测试准确率
torch.cuda.synchronize()  # 同步GPU操作（确保计时准确）
start = time.time()  # 记录训练开始时间

for epoch in range(EPOCH):  # 遍历训练轮数
    # 第step批次，高光谱数据（B,nc,11,11），激光雷达数据(B,1,11,11)，标签(64)
    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):  # 遍历训练批次

        # 将批次数据移到GPU
        b_x1 = b_x1.cuda()  # HSI批次数据
        b_x2 = b_x2.cuda()  # LiDAR批次数据
        b_y = b_y.cuda()  # 批次标签
        LiDAR_seg = torch.tensor(LiDAR_seg).float().cuda()  # LiDAR超像素数据移到GPU

        # 模型前向传播：输入HSI、LiDAR、LiDAR超像素数据，输出三个分支的分类结果
        out1, out2, out3 = model(b_x1, b_x2, LiDAR_seg)
        # 计算每个分支的损失（交叉熵损失）
        loss1 = loss_func(out1, b_y)
        loss2 = loss_func(out2, b_y)
        loss3 = loss_func(out3, b_y)
        loss = loss1 + loss2 + loss3  # 总损失（三个分支损失之和）

        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度（避免累积）
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新模型参数

        # step==0的时候进行评估，意思是，每个epoch开始的时刻进行评估，step_max=44，遍历完44后进行下一轮epoch
        if step % 50 == 0:
            model.eval()  # 切换模型为评估模式（关闭Dropout/BatchNorm训练行为）

            # 训练集验证（可选，此处仅前向传播）
            temp1 = TrainPatch1.cuda()
            temp2 = TrainPatch2.cuda()
            temp3, temp4, temp5 = model(temp1, temp2, LiDAR_seg)

            # 测试集预测（分批处理：避免测试数据量过大导致GPU内存不足）
            pred_y = np.empty((len(TestLabel)), dtype='float32')  # 存储测试集预测结果
            number = len(TestLabel) // 5000  # 计算分批数（每批5000个样本） =2

            # 分批处理测试数据
            for i in range(number):
                temp = TestPatch1[i * 5000:(i + 1) * 5000, :, :, :].cuda()  # HSI测试批次
                temp1 = TestPatch2[i * 5000:(i + 1) * 5000, :, :, :].cuda()  # LiDAR测试批次
                # 模型前向传播：三个分支结果求和（融合预测）
                temp2 = model(temp, temp1, LiDAR_seg)[2] + model(temp, temp1, LiDAR_seg)[1] + model(temp, temp1, LiDAR_seg)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()  # 取预测概率最大的类别（argmax）
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()  # 结果移回CPU并存储
                del temp, temp1, temp2, temp3  # 释放GPU内存

            # 处理剩余不足5000的样本
            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch1[(i + 1) * 5000:len(TestLabel), :, :, :].cuda()
                temp1 = TestPatch2[(i + 1) * 5000:len(TestLabel), :, :, :].cuda()
                temp2 = model(temp, temp1, LiDAR_seg)[2] + model(temp, temp1, LiDAR_seg)[1] + model(temp, temp1, LiDAR_seg)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
                del temp, temp1, temp2, temp3  # 释放GPU内存

            # 计算测试准确率
            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy, '| ')

            # 保存最佳模型（基于测试准确率）
            if accuracy > BestAcc:
                torch.save(model.state_dict(), 'BestAcc.pkl')  # 保存模型参数
                BestAcc = accuracy  # 更新最佳准确率
            model.train()  # 切换回训练模式

# ===================== 训练结束：评估最佳模型 =====================
print('Best test acc:', BestAcc)
torch.cuda.synchronize()
end = time.time()
Train_time = end - start  # 计算总训练时间
print('Total training time:', Train_time)

# 加载最佳模型参数
model.load_state_dict(torch.load('BestAcc.pkl'))
model.eval()  # 切换为评估模式
torch.cuda.synchronize()
start = time.time()  # 记录测试开始时间

# 测试集最终预测（分批处理）
pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel) // 5000
for i in range(number):
    temp = TestPatch1[i * 5000:(i + 1) * 5000, :, :].cuda()
    temp1 = TestPatch2[i * 5000:(i + 1) * 5000, :, :].cuda()
    temp2 = model(temp, temp1, LiDAR_seg)[2] + model(temp, temp1, LiDAR_seg)[1] + model(temp, temp1, LiDAR_seg)[0]
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
    del temp, temp2, temp3

# 处理剩余样本
if (i + 1) * 5000 < len(TestLabel):
    temp = TestPatch1[(i + 1) * 5000:len(TestLabel), :, :].cuda()
    temp1 = TestPatch2[(i + 1) * 5000:len(TestLabel), :, :].cuda()
    temp2 = model(temp, temp1, LiDAR_seg)[2] + model(temp, temp1, LiDAR_seg)[1] + model(temp, temp1, LiDAR_seg)[0]
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

# 转换预测结果格式并计算评价指标
pred_y = torch.from_numpy(pred_y).long()
OA, AA, Kappa, CA = output_metric(TestLabel1, pred_y)  # OA(总体准确率)、AA(平均类别准确率)、Kappa(一致性系数)、CA(各类别准确率)

# 逐类别计算准确率（验证output_metric结果）
Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))
for i in range(len(Classes)):
    cla = Classes[i]
    right = 0  # 正确预测数
    sum = 0  # 该类别总样本数
    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1
    EachAcc[i] = right.__float__() / sum.__float__()  # 该类别准确率

# 计算测试时间
torch.cuda.synchronize()
end = time.time()
Test_time = end - start

# 打印最终评估结果
print('Each class accuracy:', EachAcc)
print('The OA is: ', OA)  # 总体准确率
print('The AA is: ', AA)  # 平均类别准确率
print('The Kappa is: ', Kappa)  # Kappa系数（衡量分类一致性）
print('The Training time is: ', Train_time)  # 总训练时间
print('The Test time is: ', Test_time)  # 总测试时间
print()