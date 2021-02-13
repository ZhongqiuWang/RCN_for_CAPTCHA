import numpy as np
import cv2 as cv
from src.preprocess import Preproc
from src.crop_img import inverse_pad_resize_process


def find_whole_bu_msg(parsing_img):
    """
    找到自底向上过程中完整的输入消息
    :param parsing_img: 输入的待解析原始图片, 图片原始尺寸250x50, 即shape=(50, 250)
    :return: whole_bu_msg  16x200x1000的np.ndarray
    """
    # 反色处理, 并放大为(200, 1000), 要与crop_img中的剪裁前预处理过程一致
    scale = 4  # crop_img中原始图片放大倍数
    img = cv.resize(inverse_color(parsing_img), (scale * parsing_img.shape[1], scale * parsing_img.shape[0]))

    preproc_layer = Preproc(filter_scale=2., cross_channel_pooling=True)
    whole_bu_msg = preproc_layer.fwd_infer(img, brightness_diff_threshold=18.)
    # whole_bu_msg格式: 16x200x1000的np.ndarray, 每个位置上的值为-1或1

    whole_bu_msg[whole_bu_msg == -1] = 0    # 将whole_bu_msg中的-1全部换成0

    return whole_bu_msg


def find_whole_td_msg(candidates):
    """
    找到对于每个候选目标自顶向下过程中的回溯结果
    :param candidates: 20x6的np.ndarray, 表示20个候选目标
                       每个候选目标是一个6元数组
                       第0-3个元素分别表示 idx, score, backtrace_positions, pools_num
                       第4-5个元素表示该窗口对应原图的第几行、第几列
    :return: whole_td_msg  20x16x200x1000的np.ndarray, dtype=np.uint8
                           每个候选目标的回溯结果是16x200x1000的np.ndarray, 总共有20个
    """
    candidates_num = candidates.shape[0]
    whole_td_msg = np.zeros((candidates_num, 16, 300, 1100), dtype=np.int8)    # 原图是padding为(300, 1100)后剪裁的

    # 剪裁过程中的滑动窗口信息
    step_w = 25  # 窗口水平方向滑动步长
    step_h = 25  # 窗口竖直方向滑动步长

    for i, candidate in enumerate(candidates):
        row = candidate[4]
        col = candidate[5]
        for f, r, c in candidate[2]:
            relative_row, relative_col = inverse_pad_resize_process(r, c)
            absolute_row = relative_row + row*step_h
            absolute_col = relative_col + col*step_w
            whole_td_msg[i, f, absolute_row, absolute_col] += 1

    whole_td_msg = whole_td_msg[:, :, 50:250, 50:1050]    # 去掉padding的部分

    # 可视化td_msg
    show_whole_td_msg(whole_td_msg)

    return whole_td_msg


def inverse_color(image):
    height, width = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            img2[i, j] = (255-image[i, j])
    return img2


def show_whole_td_msg(whole_td_msg):
    """
    显示完整的td_msg
    :param whole_td_msg: np.ndarray, shape=(20, 16, 200, 1000)
    :return:
    """
    img_path = 'tmp/td_msg'
    for tag in range(20):
        final = np.zeros((200, 1000), dtype=np.uint8)
        for i in range(16):
            pic = np.zeros((200, 1000), dtype=np.uint8)
            for r, bur in enumerate(whole_td_msg[tag, i]):
                for c, buc in enumerate(bur):
                    if buc >= 1:
                        pic[r, c] += buc
            final += pic
        for r, bur in enumerate(final):
            for c, buc in enumerate(bur):
                final[r, c] = int(255 * buc)
        img_name = img_path+'/'+str(tag)+'.bmp'
        cv.imwrite(img_name, final)
