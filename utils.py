import numpy as np
from sklearn.metrics import confusion_matrix


def cal_results(matrix):
    """
    基于混淆矩阵计算分类任务的核心评价指标：OA、AA_mean、Kappa、各类别AA

    Parameters:
        matrix (np.ndarray): 混淆矩阵（shape[类别数, 类别数]），matrix[i,j]表示真实类别为i、预测类别为j的样本数

    Returns:
        OA (float): 总体准确率（Overall Accuracy）
        AA_mean (float): 平均类别准确率（Average Accuracy）
        Kappa (float): Kappa系数（衡量分类一致性）
        AA (np.ndarray): 每个类别的单独准确率（shape[类别数,]）
    """
    shape = np.shape(matrix)  # 获取混淆矩阵的形状（[类别数, 类别数]）
    number = 0  # 统计正确分类的总样本数（混淆矩阵对角线元素之和）
    sum = 0  # 计算Kappa系数时的中间变量（用于计算pe）
    AA = np.zeros([shape[0]], dtype=np.float64)  # 存储每个类别的单独准确率

    # 遍历每个类别，计算各类别准确率和中间变量
    for i in range(shape[0]):
        number += matrix[i, i]  # 累加对角线元素（正确分类的样本数）
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])  # 第i类的准确率：正确数/真实第i类的总样本数
        # 计算Kappa的pe项：(第i类真实样本数 × 第i类预测样本数)的累加
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])

    OA = number / np.sum(matrix)  # 总体准确率：总正确数/总样本数
    AA_mean = np.mean(AA)  # 平均类别准确率：所有类别准确率的均值
    pe = sum / (np.sum(matrix) ** 2)  # Kappa的期望随机一致性（pe）
    Kappa = (OA - pe) / (1 - pe)  # Kappa系数计算

    return OA, AA_mean, Kappa, AA


def output_metric(tar, pre):
    """
    输入真实标签和预测标签，生成混淆矩阵并计算分类评价指标

    Parameters:
        tar (np.ndarray/torch.Tensor): 真实标签（shape[样本数,]）
        pre (np.ndarray/torch.Tensor): 预测标签（shape[样本数,]）

    Returns:
        OA (float): 总体准确率
        AA_mean (float): 平均类别准确率
        Kappa (float): Kappa系数
        AA (np.ndarray): 每个类别的单独准确率
    """
    # 生成混淆矩阵：行=真实类别，列=预测类别
    matrix = confusion_matrix(tar, pre)
    # 调用cal_results计算评价指标
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
