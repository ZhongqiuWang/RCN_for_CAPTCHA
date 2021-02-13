import shelve
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from functools import partial

from src.crop_img import sliding_windows_cropping
from src.inference import test_image, fwd_pass_detect, bwd_pass_backtrace
from src.choose_candidates import choose_candidates, non_max_suppression_without_bwd
from src.find_msg import find_whole_bu_msg, find_whole_td_msg
from src.multiple_objects_parsing import MultiObjectsParsing


class ParseSingleCaptcha(object):
    """

    """
    def __init__(self, test_img, all_model_factors, pool_shape, is_show_tmp):
        # parse_single_captcha(test_img, all_model_factors, pool_shape)
        parse_result, score, score_terms, candidates = parse_single_captcha(test_img, all_model_factors,
                                                                            pool_shape, is_show_tmp)
        self.parse_result = parse_result
        self.score = score
        self.score_terms = score_terms
        candidates_idxs = candidates[:, 0]
        candidates_labels = []
        for candidate_idx in candidates_idxs:
            candidate_category = idx2category(candidate_idx)
            candidates_labels.append(candidate_category)
        self.candidates_labels = candidates_labels


def parse_single_captcha(test_img, all_model_factors, pool_shape, is_show_tmp, more_accurate=False):
    """
    解析单张图片
    :param test_img: 待解析图片
    :param all_model_factors: RCN模型参数
    :param pool_shape: 池节点感受野
    :param is_show_tmp:
    :param more_accurate:
    :return:
    """
    # 滑动窗口剪裁图片
    cropped_imgs = sliding_windows_cropping(test_img)
    # 剪裁结果, 一个list, 其中每个元素为一个三元tuple, tuple中第0个元素为剪裁出来的图片, 二维np.ndarray
    #                                                     第1、2个元素表示该窗口对应原图的第几行、第几列

    if more_accurate:
        # 多进程识别各剪裁出的图片
        pool = Pool(None)  # 使用num_workers个工作进程数, 为None时使用个数等于本地CPU数
        test_partial = partial(test_image, model_factors=all_model_factors, pool_shape=pool_shape)
        test_results = pool.map_async(test_partial, [d[0] for d in cropped_imgs]).get(999999)
        # test_results是一个长度为3x35=105的list, 每个元素为tuple, (winner_idx, winner_score, win_pos)
        pool.close()    # 也不知道有没有用

        # 临时将test_results保存到本地, 这样编写后面代码时不用再每次等好久好久好久
        data_dir = 'D:/WZQ/Project/PythonProject/RCN/RCN_for_CAPTCHA/first_test_results_full'
        save_test_results_to_local(test_results, data_dir)
        # 读入保存的各窗口测试结果进行下面的操作
        test_results = load_test_results_from_local(data_dir)

        # test_results是一个长度为3x35=105的list, 每个元素为tuple, (winner_idx, winner_score, win_pos)
        test_res_array = list2array(test_results, all_model_factors)
        # test_res_array格式: 3x35x6的np.ndarray, 3x35中各代表一个滑动窗口的识别结果
        #                                        每个结果是一个长度为6的ndarray
        #                                        [idx, score, backtrace_positions, pools_num, row, col]
        #                                        idx: np.int64;  score: np.float64;  backtrace_positions: Nx3 np.ndarray
        # 非极大值抑制与阈值处理选择候选目标
        candidates_num = 20  # 筛选出的候选目标的个数
        candidates = choose_candidates(test_res_array, candidates_num)
        # candidates格式: 20x6的np.ndarray, 表示筛选后的20个候选目标
        #                                  每个候选目标是一个6元数组
        #                                  第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
        #                                  第4-5个元素表示该窗口对应原图的第几行、第几列
    else:
        # 仅用前馈过程多进程识别各剪裁出的图片, 返回结果仅为一个识别结果和分数
        pool = Pool(None)  # 使用num_workers个工作进程数, 为None时使用个数等于本地CPU数
        test_partial = partial(fwd_pass_detect, model_factors=all_model_factors, pool_shape=pool_shape)
        detect_results = pool.map_async(test_partial, [d[0] for d in cropped_imgs]).get(999999)
        # test_results是一个长度为3x35=105的list, 每个元素为tuple, (idx, score)
        pool.close()  # 也不知道有没有用
        # test_results是一个长度为3x35=105的list, 每个元素为tuple, (idx, score)
        detect_res_array = fwd_pass_detect_list2array(detect_results)
        # detect_res_array格式: 3x35x4的np.ndarray, 3x35中各代表一个滑动窗口的识别结果
        #                                          每个结果是一个长度为4的ndarray
        #                                          [idx, score, row, col]
        #                                          idx: np.int64;  score: np.float64

        # 非极大值抑制选择候选目标
        candidates_num = 20  # 筛选出的候选目标的个数
        non_max_suppression_res = non_max_suppression_without_bwd(detect_res_array, candidates_num)
        # non_max_suppression_res: 20x4的np.ndarray, 表示筛选后的20个候选目标
        #                                            每个候选目标是一个4元数组
        #                                            第0-1个元素分别表示 idx, score
        #                                            第2-3个元素表示该窗口对应原图的第几行、第几列
        #                                            都是np.float64类型数据
        # 对剩下的20张图片都进行bwd过程
        non_max_suppression_res_imgs = []
        for i in range(non_max_suppression_res.shape[0]):
            img_index = int(35 * non_max_suppression_res[i, 2] + non_max_suppression_res[i, 3])
            non_max_suppression_res_imgs.append((cropped_imgs[img_index][0], int(non_max_suppression_res[i, 0])))
            # 表示idx的np.float64类型数据要转为int
        # 多进程bwd_pass_backtrace剩下的20张图片
        pool = Pool(None)  # 使用num_workers个工作进程数, 为None时使用个数等于本地CPU数
        test_partial = partial(test_image, model_factors=all_model_factors, pool_shape=pool_shape)
        test_results = pool.map_async(test_partial, [d[0] for d in non_max_suppression_res_imgs]).get(999999)
        # test_results是一个长度为20的list, 每个元素为tuple, (winner_idx, winner_score, win_pos)
        pool.close()  # 也不知道有没有用
        candidates = bwd_pass_list2array(test_results, non_max_suppression_res, all_model_factors)
        # candidates格式: 20x6的np.ndarray, 表示筛选后的20个候选目标
        #                                  每个候选目标是一个6元数组
        #                                  第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
        #                                  第4-5个元素表示该窗口对应原图的第几行、第几列, int

    if is_show_tmp:
        show_candidates(cropped_imgs, candidates)

    # 临时将candidates保存到本地, 这样编写后面代码时不用再每次等好久好久好久
    data_dir = 'tmp/tmp_test_results_full'
    save_test_results_to_local(candidates, data_dir)
    # 读入保存的各窗口测试结果进行下面的操作
    candidates = load_test_results_from_local(data_dir)

    """
    """
    # 场景解译
    all_bu_msg = find_whole_bu_msg(test_img)
    # all_bu_msg: 完整的自底向上消息, 格式: 16x200x1000的np.ndarray, 每个位置上的值为0或1
    all_td_msg = find_whole_td_msg(candidates)
    # all_td_msg: 完整的自顶向下消息, 格式: 20x16x200x1000的np.ndarray, 每个位置上的值为0或1
    parsing_result = MultiObjectsParsing(all_bu_msg, all_td_msg, candidates)
    # 解译结果

    if is_show_tmp:
        show_whole_bu_msg(all_bu_msg)
        show_parsing_res(all_td_msg, parsing_result.v)

    # 将解译结果转换为对应字符的list
    parsing_result_str_list = []
    for i, res in enumerate(parsing_result.v):
        if res == 1:
            parsing_result_str_list.append(idx2category(candidates[i, 0]))

    return parsing_result_str_list, parsing_result.score, parsing_result.score_terms, candidates


def list2array(test_list, all_model_factors):
    """
    仅限于将识别BotDetect程序parse_single_captcha()中长度为105的test_results转换为3x35的np.ndarray, 并加上对应滑动窗口行列信息
    :param test_list: 长度为105的test_results
    :param all_model_factors: RCN模型信息, 用来添加候选目标对应池节点数量信息
    :return: test_array 3x35x6的np.ndarray
    """
    tmp = [[], [], []]
    for i in range(3):
        tmp[i] = test_list[i*35:(i+1)*35]
        for j in range(35):
            tmp[i][j] = list(tmp[i][j])
            idx = tmp[i][j][0]
            tmp[i][j].append(all_model_factors[0][idx].shape[0])    # 添加 该识别结果对应池节点的个数 信息
            tmp[i][j].append(i)
            tmp[i][j].append(j)
    test_array = np.array(tmp)
    # dtype=object, type(test_array[0, 0, 5])为int
    return test_array


def fwd_pass_detect_list2array(detect_results_list):
    """

    :param detect_results_list:
    :return:
    """
    tmp = [[], [], []]
    for i in range(3):
        tmp[i] = detect_results_list[i*35:(i+1)*35]
        for j in range(35):
            tmp[i][j] = list(tmp[i][j])
            tmp[i][j].append(i)
            tmp[i][j].append(j)
    detect_res_array = np.array(tmp)
    # tmp[i][j]为list, 其中都是数, 且有float型, np.array()后原本的int保存np.float64
    return detect_res_array


def bwd_pass_list2array(test_results, non_max_suppression_res, all_model_factors):
    """

    :param test_results: 长度为20的test_results
    :param non_max_suppression_res: non_max_suppression_res: 20x4的np.ndarray, 表示筛选后的20个候选目标
                                                             每个候选目标是一个4元数组
                                                             第0-1个元素分别表示 idx, score
                                                             第2-3个元素表示该窗口对应原图的第几行、第几列
    :param all_model_factors: RCN模型信息, 用来添加候选目标对应池节点数量信息
    :return:
    """
    candidates = []
    for i in range(len(test_results)):
        tmp = list(test_results[i])
        idx = tmp[0]
        tmp.append(all_model_factors[0][idx].shape[0])  # 添加 该识别结果对应池节点的个数 信息
        tmp.append(int(non_max_suppression_res[i, 2]))  # 因为non_max_suppression_res[i, 2]原本不是int
        tmp.append(int(non_max_suppression_res[i, 3]))
        candidates.append(tmp)
    candidates = np.array(candidates)
    return candidates


def idx2category(idx):
    """
    将idx转换为对应类别
    """
    num = 24  # 每一类有24张训练图片
    index = ['3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'H', 'J', 'K', 'M', 'N', 'O', 'P',
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return index[idx // num]


def show_candidates(cropped_imgs, candidates):
    candidates_dir = 'tmp/candidates'
    for i, candidate in enumerate(candidates):
        img_name = str(i)+'_'+idx2category(candidate[0])+'_'+str(candidate[1])+'.bmp'
        img_path = candidates_dir + '/' + img_name
        cv.imwrite(img_path, cropped_imgs[candidate[4]*35+candidate[5]][0])


def save_test_results_to_local(test_results, data_dir):
    file = shelve.open(data_dir)
    data_key = "my_data"
    file[data_key] = test_results
    file.close()


def load_test_results_from_local(model_dir):
    file = shelve.open(model_dir)
    data_key = "my_data"
    model_factors = file[data_key]
    file.close()
    return model_factors


def show_whole_bu_msg(whole_bu_msg):
    """
    显示完整的bu_msg
    :param whole_bu_msg: np.ndarray, shape=(16, 200, 1000)
    :return:
    """
    final = np.zeros((200, 1000), dtype=np.uint8)
    for i in range(16):
        pic = np.zeros((200, 1000), dtype=np.uint8)
        for r, bur in enumerate(whole_bu_msg[i]):
            for c, buc in enumerate(bur):
                if buc == 1:
                    pic[r, c] += 1
        final += pic
    max_val = np.max(final)
    # print('BU', max_val)
    for r, bur in enumerate(final):
        for c, buc in enumerate(bur):
            if buc > 0:
                final[r, c] = int(255*buc/max_val)
    # cv.imshow('B', final)
    # cv.waitKey(0)
    img_path = 'tmp/test_bu_msg.bmp'
    cv.imwrite(img_path, final)
    return final


def show_parsing_res(all_td_msg, v):
    """
    显示完整的parsing_res
    :param all_td_msg: 完整的自顶向下消息, 格式: 20x16x200x1000的np.ndarray, 每个位置上的值为0或1
    :param v:
    :return:
    """
    chosen_td_msg = np.zeros((16, 200, 1000))
    for i, val in enumerate(v):
        if val == 1:
            chosen_td_msg += all_td_msg[i]
    # 以上得到chosen_td_msg, 表示根据当前的向量v确定的Td_msg, shape=(16, 200, 1000)
    final = np.zeros((200, 1000), dtype=np.uint8)
    for i in range(16):
        pic = np.zeros((200, 1000), dtype=np.uint8)
        for r, bur in enumerate(chosen_td_msg[i]):
            for c, buc in enumerate(bur):
                if buc == 1:
                    pic[r, c] += 1
        final += pic
    max_val = np.max(final)
    # print('TD', max_val)
    for r, bur in enumerate(final):
        for c, buc in enumerate(bur):
            if buc > 0:
                final[r, c] = int(255*buc/max_val)
    # cv.imshow('T', final)
    # cv.waitKey(0)
    img_path = 'tmp/parsing_res.bmp'
    cv.imwrite(img_path, final)
    return final
