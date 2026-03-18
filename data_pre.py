# 导入必要的库
from scipy import io  # 用于读取MATLAB格式的.mat数据文件
import numpy as np  # 用于数组运算和数据处理
from sklearn.decomposition import PCA  # 用于主成分分析（降维）
from sklearn.model_selection import train_test_split  # 导入但未使用，可能预留用于数据集划分

# -------------------------- 全局参数设置 --------------------------
patchsize1 = 11  # HSI数据提取的补丁（Patch）大小（11×11像素）
patchsize2 = 11  # LiDAR数据提取的补丁（Patch）大小（11×11像素）
# 计算HSI数据的填充宽度：补丁大小的一半（向下取整），用于边缘像素的补丁提取
pad_width = np.floor(patchsize1 / 2)
pad_width = np.int32(pad_width)  # 转换为整数类型（11//2=5，故填充宽度为5）
# 计算LiDAR数据的填充宽度（与HSI相同，因补丁大小一致）
pad_width2 = np.floor(patchsize2 / 2)
pad_width2 = np.int32(pad_width2)


def data_load(name="Houston"):
    """
    加载指定数据集的HSI（高光谱图像）、LiDAR（激光雷达）数据及训练/测试标签
    支持三种数据集：Houston、Trento、Augsburg，每种数据集的文件路径和变量名不同

    Parameters:
        name (str): 数据集名称，默认"Houston"
    Returns:
        Data (np.ndarray): HSI数据，shape为[高度, 宽度, 光谱波段数]，float32类型
        Data2 (np.ndarray): LiDAR数据，shape为[高度, 宽度]（单通道），float32类型
        TrLabel (np.ndarray): 训练标签矩阵，shape为[高度, 宽度]，非零值为类别标签
        TsLabel (np.ndarray): 测试标签矩阵，shape为[高度, 宽度]，非零值为类别标签
    """
    if name == "Houston":
        # Houston2013数据集文件路径
        DataPath1 = './Houston2013/HSI.mat'  # HSI数据路径
        DataPath2 = './Houston2013/LiDAR.mat'  # LiDAR数据路径
        TRPath = './Houston2013/TRLabel.mat'  # 训练标签路径
        TSPath = './Houston2013/TSLabel.mat'  # 测试标签路径

        # 读取训练/测试标签（MATLAB文件中变量名分别为TRLabel、TSLabel）
        TrLabel = io.loadmat(TRPath)['TRLabel']
        TsLabel = io.loadmat(TSPath)['TSLabel']

        # 读取HSI数据（MATLAB文件中变量名为HSI），转换为float32以节省内存并适配模型
        Data = io.loadmat(DataPath1)['HSI'].astype(np.float32)
        # 读取LiDAR数据（MATLAB文件中变量名为LiDAR），转换为float32
        Data2 = io.loadmat(DataPath2)['LiDAR'].astype(np.float32)

    elif name == "Trento":
        # Trento数据集文件路径（存储在dataset子目录下）
        DataPath1 = './Trento/HSI.mat'
        DataPath2 = './Trento/LiDAR.mat'
        TRPath = './Trento/TRLabel.mat'
        TSPath = './Trento/TSLabel.mat'

        # 读取标签（变量名与Houston一致）
        TrLabel = io.loadmat(TRPath)['TRLabel']
        TsLabel = io.loadmat(TSPath)['TSLabel']
        # 读取数据并转换类型
        Data = io.loadmat(DataPath1)['HSI'].astype(np.float32)
        Data2 = io.loadmat(DataPath2)['LiDAR'].astype(np.float32)

    elif name == "Augsburg":
        # Augsburg数据集文件路径（数据变量名与前两个数据集不同）
        DataPath1 = './dataset/Augsburg/data_DSM.mat'  # LiDAR（DSM数字表面模型）数据路径
        DataPath2 = './dataset/Augsburg/data_HS_LR.mat'  # HSI（低分辨率）数据路径
        TRPath = './dataset/Augsburg/TrainImage.mat'  # 训练标签路径（变量名：TrainImage）
        TSPath = './dataset/Augsburg/TestImage.mat'  # 测试标签路径（变量名：TestImage）

        # 读取标签（变量名与前两个数据集不同）
        TrLabel = io.loadmat(TRPath)['TrainImage']
        TsLabel = io.loadmat(TSPath)['TestImage']
        # 读取数据（注意变量名对应：HSI为data_HS_LR，LiDAR为data_DSM）
        Data = io.loadmat(DataPath1)['data_HS_LR'].astype(np.float32)
        Data2 = io.loadmat(DataPath2)['data_DSM'].astype(np.float32)

    return Data, Data2, TrLabel, TsLabel


def nor_pca(Data, Data2, ispca=True):
    """
    对HSI和LiDAR数据进行归一化处理，并对HSI数据可选进行PCA降维
    目的：消除量纲影响，减少HSI数据冗余，提升模型训练效率

    Parameters:
        Data (np.ndarray): 原始HSI数据，shape[高度, 宽度, 波段数]
        Data2 (np.ndarray): 原始LiDAR数据，shape[高度, 宽度]
        ispca (bool): 是否对HSI进行PCA降维，默认True
    Returns:
        PC (np.ndarray): 归一化后（或PCA降维后）的HSI数据，shape[高度, 宽度, 新波段数]
        Data2 (np.ndarray): 归一化后的LiDAR数据
        NC (int): 处理后HSI的波段数（PCA后为20，否则为原始波段数）
    """
    [m, n, l] = Data.shape  # m：图像高度，n：图像宽度，l：原始HSI波段数
    # 对HSI每个波段单独做min-max归一化（映射到[0,1]区间）
    for i in range(l):
        minimal = Data[:, :, i].min()  # 第i波段的最小值
        maximal = Data[:, :, i].max()  # 第i波段的最大值
        Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)  # 归一化公式

    # 对LiDAR数据做全局min-max归一化（单通道，直接全局计算）
    minimal = Data2.min()
    maximal = Data2.max()
    Data2 = (Data2 - minimal) / (maximal - minimal)

    # 若开启PCA，对HSI进行降维（保留20个主成分，解释大部分方差）
    if ispca:
        NC = 20  # PCA保留的主成分数（可调整）
        # 将HSI从[H,W,C]重塑为[H*W,C]（样本数×特征数），适配PCA输入格式
        PC = np.reshape(Data, (m * n, l))
        # 初始化PCA模型（n_components=20：保留20个主成分，copy=True：不修改原始数据）
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)  # 拟合并转换数据
        PC = np.reshape(PC, (m, n, NC))  # 重塑回[H,W,NC]格式
    else:
        NC = l  # 不做PCA，波段数保持原始数量
        PC = Data  # 直接返回归一化后的HSI

    return PC, Data2, NC


def border_inter(PC, Data2, NC):
    """
    对PCA处理后的HSI和归一化后的LiDAR数据进行边界填充
    目的：解决图像边缘像素无法提取完整补丁（Patch）的问题（边缘像素无足够邻域）

    Parameters:
        PC (np.ndarray): PCA后/归一化后的HSI数据，shape[H,W,NC]
        Data2 (np.ndarray): 归一化后的LiDAR数据，shape[H,W]
        NC (int): HSI的波段数
    Returns:
        x (np.ndarray): 填充后的HSI数据，shape[H+2*pad_width, W+2*pad_width, NC]
        x2 (np.ndarray): 填充后的LiDAR数据，shape[H+2*pad_width2, W+2*pad_width2]
    """
    # 以HSI第一个波段为例，确定填充后的图像尺寸（所有波段填充方式一致）
    temp = PC[:, :, 0]
    # 对称填充：以边缘为对称轴，镜像填充（避免边缘信息失真，优于零填充）
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape  # 填充后的图像尺寸（H+2*5, W+2*5）

    # 初始化填充后的HSI数组（数据类型float32）
    x = np.empty((m2, n2, NC), dtype='float32')
    # 对HSI每个波段分别进行对称填充（保证每个波段的边缘处理一致）
    for i in range(NC):
        temp = PC[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2

    # 对LiDAR数据进行对称填充（单通道，直接填充）
    x2 = Data2
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2

    return x, x2


def con_data(x, x2, TrLabel, TsLabel, NC):
    """
    从填充后的HSI和LiDAR数据中，提取训练集和测试集的补丁（Patch）及对应标签
    补丁提取规则：围绕每个有标注（非零）的像素，提取固定大小的邻域作为输入样本

    Parameters:
        x (np.ndarray): 填充后的HSI数据，shape[H_pad, W_pad, NC]
        x2 (np.ndarray): 填充后的LiDAR数据，shape[H_pad2, W_pad2]
        TrLabel (np.ndarray): 训练标签矩阵，shape[H,W]
        TsLabel (np.ndarray): 测试标签矩阵，shape[H,W]
        NC (int): HSI的波段数
    Returns:
        TrainPatch (np.ndarray): HSI训练补丁，shape[训练样本数, NC, patchsize1, patchsize1]
        TestPatch (np.ndarray): HSI测试补丁，shape[测试样本数, NC, patchsize1, patchsize1]
        TrainPatch2 (np.ndarray): LiDAR训练补丁，shape[训练样本数, 1, patchsize2, patchsize2]
        TestPatch2 (np.ndarray): LiDAR测试补丁，shape[测试样本数, 1, patchsize2, patchsize2]
        TrainLabel (np.ndarray): 训练样本标签，shape[训练样本数]
        TestLabel (np.ndarray): 测试样本标签，shape[测试样本数]
        TrainLabel2 (np.ndarray): 与TrainLabel一致（冗余，用于对齐数据结构）
        TestLabel2 (np.ndarray): 与TestLabel一致（冗余，用于对齐数据结构）
    """
    # -------------------------- 提取HSI训练补丁 --------------------------
    # 找到训练标签中非零的像素坐标（这些是有标注的训练样本）
    [ind1, ind2] = np.where(TrLabel != 0)  # ind1：行索引，ind2：列索引
    TrainNum = len(ind1)  # 训练样本总数2832
    # 初始化HSI训练补丁数组：[样本数, 通道数, 高度, 宽度]（适配深度学习模型输入格式：NCHW）
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)  # 存储每个训练补丁的标签

    # 计算填充后图像中对应的坐标（原始坐标 + 填充宽度，因填充后图像整体偏移）
    ind3 = ind1 + pad_width  # 填充后的行索引
    ind4 = ind2 + pad_width  # 填充后的列索引

    # 逐个提取训练补丁
    for i in range(len(ind1)):
        # 提取以(ind3[i], ind4[i])为中心的11×11补丁（HSI）
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        # 重塑补丁：[11×11, NC] → 转置为[NC, 11×11] → 重塑为[NC, 11, 11]（NCHW格式）
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch  # 存储补丁
        TrainLabel[i] = TrLabel[ind1[i], ind2[i]]  # 存储对应标签

    # -------------------------- 提取HSI测试补丁 --------------------------
    [ind1, ind2] = np.where(TsLabel != 0)  # 找到测试标签中非零像素坐标
    TestNum = len(ind1)  # 测试样本总数
    # 初始化HSI测试补丁数组（格式与训练补丁一致）
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)

    ind3 = ind1 + pad_width  # 填充后的坐标偏移
    ind4 = ind2 + pad_width

    # 逐个提取测试补丁（逻辑与训练补丁一致）
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        TestLabel[i] = TsLabel[ind1[i], ind2[i]]

    # -------------------------- 提取LiDAR训练补丁 --------------------------
    [ind1, ind2] = np.where(TrLabel != 0)  # 复用训练标签坐标（与HSI训练样本一一对应）
    # 初始化LiDAR训练补丁数组：[样本数, 1, patchsize2, patchsize2]（单通道NCHW格式）
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)

    ind3 = ind1 + pad_width2  # LiDAR填充后的坐标偏移
    ind4 = ind2 + pad_width2

    # 逐个提取LiDAR训练补丁
    for i in range(len(ind1)):
        # 提取以(ind3[i], ind4[i])为中心的11×11补丁（LiDAR单通道）
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        # 重塑补丁：[11×11, 1] → 转置为[1, 11×11] → 重塑为[1, 11, 11]（单通道NCHW）
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        TrainLabel2[i] = TrLabel[ind1[i], ind2[i]]

    # -------------------------- 提取LiDAR测试补丁 --------------------------
    [ind1, ind2] = np.where(TsLabel != 0)  # 复用测试标签坐标
    # 初始化LiDAR测试补丁数组（格式与训练补丁一致）
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)

    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2

    # 逐个提取LiDAR测试补丁（逻辑与训练补丁一致）
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        TestLabel2[i] = TsLabel[ind1[i], ind2[i]]

    return TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2


def con_data1(x, x2, AllLabel, NC):
    """
    提取全量有标签数据的补丁（无训练/测试划分），用于全图预测或自定义数据集划分
    功能与con_data类似，但仅处理单一标签矩阵（AllLabel）

    Parameters:
        x (np.ndarray): 填充后的HSI数据，shape[H_pad, W_pad, NC]
        x2 (np.ndarray): 填充后的LiDAR数据，shape[H_pad2, W_pad2]
        AllLabel (np.ndarray): 全量标签矩阵（含所有有标注像素），shape[H,W]
        NC (int): HSI的波段数
    Returns:
        Allpatch (np.ndarray): HSI全量补丁，shape[样本数, NC, patchsize1, patchsize1]
        Allpatch2 (np.ndarray): LiDAR全量补丁，shape[样本数, 1, patchsize2, patchsize2]
    """
    # 找到全量标签中非零的像素坐标
    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)  # 样本总数（变量名TestNum为历史遗留，实际为全量样本数）
    # 初始化HSI全量补丁数组（NCHW格式）
    Allpatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)  # 标签数组（未返回，仅中间变量）

    ind3 = ind1 + pad_width  # 填充后的坐标偏移
    ind4 = ind2 + pad_width

    # 逐个提取HSI全量补丁（逻辑与con_data一致）
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        Allpatch[i, :, :, :] = patch
        TestLabel[i] = AllLabel[ind1[i], ind2[i]]

    # 提取LiDAR全量补丁
    [ind1, ind2] = np.where(AllLabel != 0)  # 复用坐标
    # 初始化LiDAR全量补丁数组（单通道NCHW格式）
    Allpatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)  # 标签数组（未返回）

    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2

    # 逐个提取LiDAR全量补丁（逻辑与con_data一致）
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        Allpatch2[i, :, :, :] = patch
        TestLabel2[i] = AllLabel[ind1[i], ind2[i]]

    return Allpatch, Allpatch2


def getIndex(TestLabel, temp):
    """
    获取测试标签中非零像素的坐标索引（转换为MATLAB风格的1-based索引）
    目的：用于结果可视化或与MATLAB结果对比（MATLAB索引从1开始，Python从0开始）

    Parameters:
        TestLabel (np.ndarray): 测试标签矩阵，shape[H,W]
        temp (int): 非零标签像素的总数（需与实际数量一致）
    Returns:
        index (np.ndarray): 1-based坐标索引，shape[2, temp]，行0为行索引，行1为列索引
    """
    # 初始化索引数组：2行（行/列），temp列（样本数），整数类型
    index = np.empty(shape=(2, temp), dtype=int)
    k = 0  # 索引计数器
    # 遍历标签矩阵的每个像素
    for i in range(len(TestLabel)):
        for j in range(len(TestLabel[0])):
            if TestLabel[i][j] != 0:  # 找到非零标签像素
                index[0][k] = i + 1  # 行索引+1（转换为1-based）
                index[1][k] = j + 1  # 列索引+1（转换为1-based）
                k += 1  # 计数器递增

    return index