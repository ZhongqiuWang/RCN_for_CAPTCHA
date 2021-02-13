import numpy as np
from functools import partial
from src.scene_scoring_function import scene_scoring


class MultiObjectsParsing(object):
    """

    """
    def __init__(self, all_bu_msg, all_td_msg, candidates):
        """
        初始化实例属性, 即得到了最终结果
        :param all_bu_msg: 完整的自底向上消息, 格式: 16x200x1000的np.ndarray, 每个位置上的值为0或1
        :param all_td_msg: 完整的自顶向下消息, 格式: 20x16x200x1000的np.ndarray, 每个位置上的值为0或1
        :param candidates: candidates格式: 20x6的np.ndarray, 表示筛选后的20个候选目标
                                       每个候选目标是一个6元数组
                                       第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
                                       第4-5个元素表示该窗口对应原图的第几行、第几列

        属性:
        v: 表示解译结果的向量, 长度为candidates_num的np.ndarray, dtype=np.uint8
           每个位置上为1时表示选择对应候选目标, 为0时表示不选择
        score:
        score_terms:
        """
        candidates_num = candidates.shape[0]
        ss_partial = partial(scene_scoring, all_bu_msg=all_bu_msg, all_td_msg=all_td_msg, candidates=candidates)
        # 计算多目标解译得分的函数仅参数v需要变化, 其余固定
        # ss_partial是关于向量v的函数, 返回当前v的情况下的多目标解译得分

        memo = dict()
        # 缓存字典
        # 格式:

        # # 手动查看某种解译得分
        # v = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
        # print(ss_partial(v))

        def prefix(m):
            """
            当取前m个候选目标时的最优解析, 返回一个(m, )的np.ndarray, dtype=np.uint8
            :param m: int, 表示取前m个候选目标
            :return:
            """
            prefix_parse_m = np.zeros((m,), dtype=np.uint8)

            if m in memo:
                prefix_parse_m = memo[m][0]
            elif m == 0:
                ss = 0
                final_terms_tmp = (0, 0, 0, 0)
                memo[m] = (prefix_parse_m, ss, final_terms_tmp)
            elif m == 1:
                prefix_parse_m = np.ones((m,), dtype=np.uint8)
                v_tmp = np.concatenate((prefix_parse_m, np.zeros((candidates_num-m,), dtype=np.uint8)))
                ss, term1, term2, term3, term4 = ss_partial(v_tmp)
                memo[m] = (prefix_parse_m, ss, (term1, term2, term3, term4))
            else:
                # m未被计算过且不是0或1
                ss_max = float('-INF')
                final_terms_tmp = (0, 0, 0, 0)
                for i in range(1, m):  # 第m个一定选择, 遍历前m-1个的最优决定选择哪个组合为到m为止的最优
                    zeros_behind_m = np.zeros((candidates_num-m,), dtype=np.uint8)
                    prefix_parse_m_tmp = propos(i, m)
                    v_tmp = np.concatenate((prefix_parse_m_tmp, zeros_behind_m))
                    ss, term1, term2, term3, term4 = ss_partial(v_tmp)
                    if ss >= ss_max:
                        prefix_parse_m = prefix_parse_m_tmp
                        ss_max = ss
                        final_terms_tmp = (term1, term2, term3, term4)
                memo[m] = (prefix_parse_m, ss_max, final_terms_tmp)
            return prefix_parse_m

        def propos(i, m):
            """
            必选第m个, 前i-1个取最优前缀, 中间的均取0, 返回一个(m, )的np.ndarray, dtype=np.uint8
            :param i:
            :param m:
            :return:
            """
            one_in_m = np.ones((1,), dtype=np.uint8)
            if i == 1:
                zeros_before_m = np.zeros((m-1,), dtype=np.uint8)
                propos_res = np.concatenate((zeros_before_m, one_in_m))
            else:
                zeros_before_m = np.zeros((m-i,), dtype=np.uint8)
                propos_res = np.concatenate((prefix(i-1), zeros_before_m, one_in_m))
            return propos_res

        ss_max_final = float('-INF')
        res = -1
        final_terms = (0, 0, 0, 0)
        for k in range(candidates_num+1):  # 遍历0到candidates_num
            prefix(k)
            if memo[k][1] > ss_max_final:
                ss_max_final = memo[k][1]
                res = k
                final_terms = memo[k][2]
        v = np.concatenate((memo[res][0], np.zeros((candidates_num-res,), dtype=np.uint8)))

        self.v = v
        # print(v)
        self.score = ss_max_final
        self.score_terms = final_terms
