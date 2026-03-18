import numpy as np  # 用于数组运算和数据处理
import matplotlib.pyplot as plt  # 用于超像素结果可视化
from skimage.segmentation import mark_boundaries  # 用于绘制超像素边界（辅助可视化）
import cv2  # 核心库，调用ximgproc模块的超像素算法（LSC/SEEDS）
import math  # 用于计算超像素初始尺寸


def LSC_superpixel(I, nseg):  # 并没有用到
    '''
    基于LSC（Linear Spectral Clustering，线性谱聚类）算法生成超像素
    LSC是高效的超像素分割算法，通过谱聚类思想在特征空间划分像素，兼顾边界贴合度和计算效率

    Parameters:
        I (np.ndarray): 输入图像，shape为[高度, 宽度, 通道数]（支持单通道/多通道数据）
        nseg (int): 期望生成的超像素数量（实际数量可能略有偏差，算法会自适应调整）
    Returns:
        np.ndarray: 超像素标签矩阵，shape与输入图像一致，每个像素值为其所属超像素的索引（int64类型）
    '''
    # 计算超像素的初始尺寸：超像素近似为正方形，尺寸≈√(图像总像素数/期望超像素数)
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))

    # 初始化LSC超像素分割器
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,  # 输入图像
        region_size=size,  # 超像素的初始区域大小（基于上面的计算）
        ratio=0.005  # 紧凑度参数（控制超像素形状规整度，值越小越贴合图像边界）
    )

    superpixelLSC.iterate()  # 执行LSC算法迭代（核心分割步骤）
    # 强制超像素连通性：过滤小于25个像素的微小超像素（避免碎片化）
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()  # 获取每个像素的超像素标签

    return np.array(segments, np.int64)  # 转换为int64类型并返回


def SegmentsLabelProcess(labels):  # 重新映射标签值，确保标签连续无空缺

    labels = np.array(labels, np.int64)  # 确保输入为int64类型
    H, W = labels.shape  # 获取图像高度和宽度
    # 提取所有唯一的标签值（去重）
    ls = list(set(np.reshape(labels, [-1]).tolist()))   # [1，3，4，5，7....]

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels  # 初始化新标签矩阵（复用内存）
    # 遍历每个像素，替换为连续标签
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SEEDS(object):
    """
    基于SEEDS（Superpixels Extracted via Energy-Driven Sampling）算法的超像素分割类
    SEEDS是一种高效的超像素算法，通过能量驱动采样生成超像素，适合实时场景，支持多模态数据适配

    Attributes:
        n_segments (int): 期望超像素数量
        num_levels (int): 金字塔层数（控制超像素分割的多尺度性，默认2）
        prior (int): 先验权重（平衡颜色相似度和空间距离，默认1）
        histogram_bins (int): 颜色直方图的bin数量（用于特征量化，默认5）
        num_iterations (int): 迭代次数（迭代越多分割越精细，默认4）
        data (np.ndarray): 输入数据（初始化时传入，此处命名为LiDAR，支持多模态数据兼容）
        segments (np.ndarray): 超像素标签矩阵（shape[高度, 宽度]）
        superpixel_count (int): 实际生成的超像素数量
        S (np.ndarray): 超像素均值矩阵（shape[超像素数, 通道数]，每行是一个超像素的平均特征）
        Q (np.ndarray): 像素-超像素关联矩阵（shape[总像素数, 超像素数]，one-hot编码，像素属于某超像素则对应位置为1）
    """

    def __init__(self, LiDAR, n_segments, num_levels=2, prior=1, histogram_bins=5, num_iterations=4):
        """
        类初始化：配置SEEDS算法参数并存储输入数据

        Parameters:
            LiDAR (np.ndarray): 输入数据（可为单通道LiDAR或多通道图像，shape[高度, 宽度, 通道数]）
            n_segments (int): 期望生成的超像素数量
            num_levels (int, optional): 金字塔层数，默认2
            prior (int, optional): 先验权重，默认1
            histogram_bins (int, optional): 颜色直方图bin数，默认5
            num_iterations (int, optional): 迭代次数，默认4
        """
        self.n_segments = n_segments  # 超像素数量
        self.num_levels = num_levels  # 层数
        self.prior = prior  # 先验权重
        self.histogram_bins = histogram_bins  # 直方图bin数
        self.num_iterations = num_iterations  # 迭代次数
        self.data = LiDAR  # 存储输入数据（支持LiDAR等多模态数据）

    def SEEDS_superpixel(self, I):   # 输入像素图，输出像素图的超像素标签图

        I_new = np.array(I[:, :, 0:3], np.float32).copy()  # [349,1095,1]
        height, width, channels = I_new.shape  # 获取图像尺寸和通道数

        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            width,  # 1095
            height,  # 349
            channels,  # 1
            int(self.n_segments),  # 期望超像素数（转换为整数）1108
            num_levels=self.num_levels,  # 层数
            prior=self.prior,  # 先验权重
            histogram_bins=self.histogram_bins  # 直方图bin数
        )

        # 执行SEEDS算法迭代（核心分割步骤，迭代次数为初始化时配置的参数）
        seeds.iterate(I_new, self.num_iterations)

        # 获取每个像素的超像素标签
        segments = seeds.getLabels()

        return segments    # (349,1905)

    def get_Q_and_S_and_Segments(self, img):
        # 输入LiDAR图(349,1905，1)，生成Q像素-超像素关联矩阵（664845，949），S超像素均值矩阵，segment（349，1905）

        (h, w, d) = img.shape  # h=高度，w=宽度，d=1
        segments = self.SEEDS_superpixel(img)  # 调用SEEDS算法生成初始超像素标签

        # 预留：若需替换为SLIC超像素算法，可取消下面注释（当前默认使用SEEDS）
        # SLIC（Simple Linear Iterative Clustering）：另一种常用超像素算法
        # segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
        #                 convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
        #                 min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        # 检查超像素标签是否连续：若最大标签+1 != 唯一标签数，说明存在空缺标签，需修正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)  # 修正为连续标签
        self.segments = segments  # 存储
        superpixel_count = segments.max() + 1  # 计算实际超像素数量949（标签从0开始，故最大标签+1）
        self.superpixel_count = superpixel_count  # 存储超像素数量
        print("superpixel_count", superpixel_count)  # 打印实际超像素数（调试用）

        # 可视化超像素边界：mark_boundaries在原始图像上绘制超像素分割线
        # 此处用img[:, :, 0]（第一通道）作为底图，兼容单通道/多通道数据
        out = mark_boundaries(img[:, :, 0], segments)
        plt.figure()  # 创建绘图窗口
        plt.imshow(out, cmap='gray')  # 显示超像素边界图（灰度图适配单通道数据）
        plt.title('Superpixel Boundaries')  # 添加标题（可选，增强可读性）
        plt.axis('off')  # 关闭坐标轴（可选，聚焦超像素效果）
        plt.show()  # 显示图像

        # 数据reshape：为后续矩阵计算做准备
        segments_flat = np.reshape(segments, [-1])  # 超像素标签展平为1维（664845）
        total_pixels = w * h  # 图像总像素数664845

        # 初始化S和Q矩阵
        S = np.zeros([superpixel_count, d], dtype=np.float32)  # 超像素均值矩阵（949，1）
        Q = np.zeros([total_pixels, superpixel_count], dtype=np.float32)  #关联矩阵（664845，949）
        img_flat = np.reshape(img, [-1, d])  # 输入数据展平为1维（shape[664945，1）

        # 遍历每个超像素，计算均值特征和关联矩阵
        for i in range(superpixel_count):
            # 找到属于第i个超像素的所有像素索引
            pixel_indices = np.where(segments_flat == i)[0]
            pixel_count = len(pixel_indices)  # 第i个超像素的像素数量
            # 提取该超像素的所有像素特征
            superpixel_pixels = img_flat[pixel_indices]
            # 计算超像素的均值特征（按通道平均）
            superpixel_mean = np.sum(superpixel_pixels, axis=0) / pixel_count
            S[i] = superpixel_mean  # 存储每个超像素的均值特征
            Q[pixel_indices, i] = 1  # 关联矩阵赋值：属于该超像素的像素对应位置设为1（one-hot）

        self.S = S  # 存储超像素均值矩阵
        self.Q = Q  # 存储像素-超像素关联矩阵

        return Q, S, self.segments  # 返回Q、S和超像素标签

    def get_A(self, sigma: float):  # 计算超像素邻接矩阵A：描述超像素之间的相邻关系和相似度

        # 初始化邻接矩阵：shape[949, 949]，初始值为0（无连接）
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape  # 获取超像素标签矩阵的尺寸（与输入图像一致）

        # 遍历图像的2×2窗口：判断窗口内是否存在不同超像素（即相邻超像素）
        # 窗口范围：i从0到h-2，j从0到w-2（避免窗口超出图像边界）
        for i in range(h - 2):
            for j in range(w - 2):
                # 提取当前2×2窗口内的超像素标签
                window_labels = self.segments[i:i + 2, j:j + 2]
                window_max = np.max(window_labels).astype(np.int32)  # 窗口内最大标签
                window_min = np.min(window_labels).astype(np.int32)  # 窗口内最小标签

                # 若窗口内存在不同标签（说明有相邻超像素）
                if window_max != window_min:
                    # 取窗口内两个不同的超像素索引（max和min，确保唯一）
                    sp_idx1 = window_max
                    sp_idx2 = window_min

                    # 若邻接矩阵已记录该对超像素的相似度，跳过（避免重复计算）
                    if A[sp_idx1, sp_idx2] != 0:
                        continue

                    # 获取两个超像素的均值特征
                    sp1_mean = self.S[sp_idx1]
                    sp2_mean = self.S[sp_idx2]
                    # 计算特征差异的平方和（欧氏距离的平方）
                    feature_diff = np.sum(np.square(sp1_mean - sp2_mean))
                    # 用高斯核计算相似度：diss = exp(-(特征差异²)/sigma²)，值越大相似度越高
                    similarity = np.exp(-feature_diff / (sigma ** 2))
                    # 邻接矩阵对称赋值（超像素i与j的相似度=j与i的相似度）
                    A[sp_idx1, sp_idx2] = A[sp_idx2, sp_idx1] = similarity

        return A


class Superpixel_Seg(object):  # 输入的是LiDAR数据[349，1905，1]

    def __init__(self, scale):  # 期望划分的超像素个数
        """
        类初始化：配置超像素尺度参数

        Parameters:
            scale (float): 超像素尺度（图像总像素数 / scale = 初始期望超像素数）
        """
        self.scale = scale  # 期望超像素包含像素数量

    def SGB(self, data):  # 输入的是LiDAR数据[349，1905，1]

        height, width = data.shape[:2]  # 获取数据的高度和宽度
        n_segments_init = height * width / self.scale  # 计算初始期望超像素数：图像总像素数 / 尺度参数scale
        print(f"Initial number of segments: {n_segments_init}")  # 打印初始期望超像素数（调试用）

        # 初始化SEEDS超像素实例：传入数据和初始期望超像素数，使用默认算法参数
        myseeds = SEEDS(
            data,
            n_segments=n_segments_init,
            num_levels=2,
            prior=1,
            histogram_bins=5,
            num_iterations=4
        )

        # 生成Q（关联矩阵）、S（均值矩阵）和超像素标签
        Q, S, Segments = myseeds.get_Q_and_S_and_Segments(data)
        # 生成超像素邻接矩阵A（sigma=10为经验值，可根据数据调整）
        A = myseeds.get_A(sigma=10)

        return Q, S, A, Segments  # 返回超像素图的核心组件