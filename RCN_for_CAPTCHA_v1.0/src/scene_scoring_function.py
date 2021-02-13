import numpy as np
from math import sqrt
from scipy.ndimage.morphology import grey_dilation


def scene_scoring(v, all_bu_msg, all_td_msg, candidates, overlap_penalty=-1.5, delta=-0.9, gain=1.2, base=0):
    """
    计算某个确定的v的情况下多目标解译的得分
    :param v: 表示解译结果的向量, (candidates_num, )的np.ndarray, dtype=np.uint8, 每个位置上为1时表示选择对应候选目标
    :param all_bu_msg: 完整的自底向上消息, 格式:    16x200x1000的np.ndarray, 每个位置上的值为0或1
    :param all_td_msg: 完整的自顶向下消息, 格式: 20x16x200x1000的np.ndarray, 每个位置上的值为0或1
    :param candidates: candidates格式: 20x6的np.ndarray, 表示筛选后的20个候选目标
                                       每个候选目标是一个6元数组
                                       第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
                                       第4-5个元素表示该窗口对应原图的第几行、第几列
    :param overlap_penalty: 重叠惩罚系数
    :param delta: 复杂度惩罚系数
    :param gain:
    :param base:
    :return: ss  多目标解译得分
    """
    ss = 0    # 表示多目标解译得分

    chosen_td_msg = np.zeros(all_bu_msg.shape)    # shape=(16, 200, 1000)
    for i, val in enumerate(v):
        if val == 1:
            chosen_td_msg += all_td_msg[i]
    # 以上得到chosen_td_msg, 表示根据当前的向量v确定的Td_msg, shape=(16, 200, 1000)
    chosen_td_msg_dil = dil_td_msg(chosen_td_msg)
    bu_multiply_td = np.multiply(all_bu_msg, chosen_td_msg_dil)
    # 矩阵对应位置元素相乘, bu_multiply_td.shape=(16, 200, 1000)
    term1 = np.sum(bu_multiply_td, axis=None)
    ss += term1
    # 第一项的结果, bu_multiply_td中所有元素之和

    chosen_td_msg_flat_dil = np.zeros((all_bu_msg.shape[1], all_bu_msg.shape[2]))
    for i, val in enumerate(v):
        if val == 1:
            chosen_td_msg_flat_dil += dil_flat(all_td_msg[i])
    # 以上得到chosen_td_msg_flat_dil, shape=(200, 1000)
    term2 = overlap_penalty * np.sum(np.maximum(0, chosen_td_msg_flat_dil-1))
    ss += term2
    # 加上第二项后的结果

    all_pools_num = candidates[:, 3]
    # all_pools_num: 每个候选目标激活的池的个数, (20, )的np.ndarray, 每个位置上是一个int型变量
    tmp = 0
    for i, val in enumerate(v):
        if val == 1:
            # tmp += sqrt(all_pools_num[i])
            tmp += 0.7*all_pools_num[i]
    term3 = delta * tmp
    ss += term3
    # 加上第三项后的结果

    # 我觉得可以再加一项, 选择得分尽量高的候选目标
    all_scores = candidates[:, 1]
    final_score = 0
    num = 0
    for i, val in enumerate(v):
        if val == 1:
            final_score += all_scores[i]
            num += 1
    term4 = gain*(final_score/num-base)
    ss += term4

    return ss, term1, term2, term3, term4


def dil_td_msg(chosen_td_msg, dilate_shape=(4, 4)):
    """

    :param chosen_td_msg:
    :param dilate_shape: (h, w)  h大一些使得膨胀在垂直方向更大
    :return:
    """
    chosen_td_msg_dil = np.zeros((16, 200, 1000), dtype=np.uint8)
    for i, td_msg in enumerate(chosen_td_msg):
        chosen_td_msg_dil[i] = grey_dilation(td_msg, dilate_shape)

    return chosen_td_msg_dil


def dil_flat(td_msg, dilate_shape=(4, 4)):
    """
    对单个候选目标的td_msg做扁平化并膨胀处理
    :param td_msg: 单个候选目标的td_msg, shape=(16, 200, 1000)
    :param dilate_shape: 膨胀处理的核的形状
    :return:
    """
    td_msg_flat = np.sum(td_msg, axis=0)
    # 将td_msg扁平化, 即按第一个维度相加, td_msg_flat.shape=(200, 1000)
    td_msg_flat_dil = grey_dilation(td_msg_flat, dilate_shape)
    return td_msg_flat_dil



