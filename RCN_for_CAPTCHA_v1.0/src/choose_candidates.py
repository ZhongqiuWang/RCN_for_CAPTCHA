import numpy as np


def choose_candidates(test_res_array, candidates_num=20):
    """
    非极大值抑制与阈值处理, 从3x35个结果中选出candidates_num个候选目标
    :param test_res_array:3x35x6的np.ndarray, 3x35中各代表一个滑动窗口的识别结果
                                              每个结果是一个长度为6的ndarray
                                              [idx, score, backtrace_positions, pools_num, row, col]
                                              idx: np.int64;  score: np.float64;  backtrace_positions: Nx3 np.ndarray
    :param candidates_num: 筛选出的候选目标的个数
    :return:
    """
    # 非极大值抑制, 从3x35个结果中选择各列得分最高的一个, 组成剩下的35个候选结果
    non_max_suppression_res = []
    for i in range(test_res_array.shape[1]):
        temp = test_res_array[:, i, 1]
        score_max = np.argsort(temp)[-1]
        non_max_suppression_res.append(test_res_array[score_max, i, :])
    non_max_suppression_res = np.array(non_max_suppression_res)
    # non_max_suppression_res格式: 35x6的np.ndarray, 表示非极大值抑制后的35个候选目标
    #                                               每个候选目标是一个6元数组
    #                                               第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
    #                                               第4-5个元素表示该窗口对应原图的第几行、第几列

    # 进一步阈值处理, 从35个候选目标中选择得分最高的candidates_num个作为候选目标
    temp = non_max_suppression_res[:, 1]
    score_top_num = np.argsort(temp)[-1*candidates_num:]
    candidates = []
    for i in np.argsort(score_top_num):
        tmp_tag = score_top_num[i]
        candidates.append(non_max_suppression_res[tmp_tag])
    candidates = np.array(candidates)

    return candidates


def non_max_suppression_without_bwd(detect_res_array, candidates_num=20):
    """
    非极大值抑制, 这里主要用于对仅进行前馈过程的识别进行处理, 从3x35个结果中每列选择1个
    :param detect_res_array: detect_res_array格式: 3x35x4的np.ndarray, 3x35中各代表一个滑动窗口的识别结果
                                                   每个结果是一个长度为4的ndarray
                                                   [idx, score, row, col]
                                                   idx: np.int64;  score: np.float64
    :param candidates_num: 筛选出的候选目标的个数
    :return:
    """
    # 非极大值抑制, 从3x35个结果中选择各列得分最高的一个, 组成剩下的35个候选结果
    non_max_suppression_res = []
    for i in range(detect_res_array.shape[1]):
        temp = detect_res_array[:, i, 1]
        score_max = np.argsort(temp)[-1]
        non_max_suppression_res.append(detect_res_array[score_max, i, :])
    non_max_suppression_res = np.array(non_max_suppression_res)
    # non_max_suppression_res格式: 35x4的np.ndarray, 表示非极大值抑制后的35个候选目标
    #                                               每个候选目标是一个4元数组
    #                                               第0-2个元素分别表示 idx, score, pools_num
    #                                               第3-4个元素表示该窗口对应原图的第几行、第几列

    # 进一步阈值处理, 从35个候选目标中选择得分最高的candidates_num个作为候选目标
    temp = non_max_suppression_res[:, 1]
    score_top_num = np.argsort(temp)[-1 * candidates_num:]
    candidates = []
    for i in np.argsort(score_top_num):
        tmp_tag = score_top_num[i]
        candidates.append(non_max_suppression_res[tmp_tag])
    candidates = np.array(candidates)

    return candidates
