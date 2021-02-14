import cv2 as cv
import numpy as np


def sliding_windows_cropping(img, is_show_tmp):
    """
    滑动窗口剪裁图片
    :param img: 被剪裁的图片, BotDetect中图片原始尺寸250x50, 即shape=(50, 250)
    :param is_show_tmp:
    :return: cropped_imgs
             剪裁结果, 一个list, 其中每个元素为一个三元tuple
             tuple中第0个元素为剪裁出来的图片, 二维np.ndarray
             第1、2个元素表示该窗口对应原图的第几行、第几列
    """
    cropped_imgs = []    # 保存滑动窗口剪裁结果的list

    # 判断是否反色
    if img[0, 0] >= 128:
        is_inverse = True
    else:
        is_inverse = False

    scale = 4    # 原始图片放大倍数
    win_w = 250    # 滑动窗口宽度
    win_h = 250    # 滑动窗口高度
    step_w = 25    # 窗口水平方向滑动步长
    step_h = 25    # 窗口竖直方向滑动步长

    # 反色处理, 并放大为(200, 1000), cv2.resize()中的尺寸为(w, h)
    if is_inverse:
        img2 = cv.resize(inverse_color(img), (scale*img.shape[1], scale*img.shape[0]))
    else:
        img2 = cv.resize(img, (scale * img.shape[1], scale * img.shape[0]))
    # padding为(300, 1100)
    img_complete = np.pad(img2, pad_width=((50, 50), (50, 50)), mode='constant', constant_values=0)

    num_w = int((img_complete.shape[1]-win_w)/step_w + 1)    # 水平方向窗口个数, 比滑动次数多1, 这里为35
    num_h = int((img_complete.shape[0]-win_h)/step_h + 1)    # 竖直方向窗口个数, 这里为3
    for i in range(num_h):
        for j in range(num_w):
            # 滑动窗口剪裁
            tmp = img_complete[i*step_h:i*step_h+win_h, j*step_w:j*step_w+win_w]

            # 剪裁后padding并resize为和训练图片相同大小
            res = pad_resize_process(tmp)

            # 剪裁结果添加到list中
            if is_show_tmp:
                save_cropped_img(res, i, j)
            cropped_imgs.append((res, i, j))

    return cropped_imgs


def inverse_color(image):
    """
    反色处理
    :param image:
    :return:
    """
    height, width = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            img2[i, j] = (255-image[i, j])
    return img2


def pad_resize_process(img):
    """
    调整剪裁出来的图片, 剪裁后的shape(250, 250), padding为(570, 570), 再resize为(200, 200)
    :param img: 剪裁出的原始patch, shape(250, 250)
    :return: 调整后的结果, shape(200, 200)
    """
    padding = 160
    img_size = (200, 200)
    tmp_pad = np.pad(img, pad_width=((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    # resize为和训练图片相同大小
    res = cv.resize(tmp_pad, img_size)
    return res


def inverse_pad_resize_process(r, c):
    """
    调整剪裁出来的图片的反过程, 计算剪裁结果(200, 200)上一点位置对应于pad_resize过程前(250, 250)上的位置
    主要用于在回溯轮廓的时候调用
    :param r: (200, 200)的patch上的坐标r
    :param c: (200, 200)的patch上的坐标c
    :return:
    """
    new_r = int(r/200*570-160)
    new_c = int(c/200*570-160)
    return new_r, new_c


def save_cropped_img(cropped_img, i, j):
    img_dir = '../tmp/sliding_windows'
    new_name = img_dir+'/'+str(i)+'_'+str(j)+'.bmp'
    cv.imwrite(new_name, cropped_img)


def show_bu_msg_whole(bu_msg):
    final = np.zeros((50, 250), dtype=np.uint8)
    for i in range(16):
        pic = np.zeros((50, 250), dtype=np.uint8)
        for r, bur in enumerate(bu_msg[i]):
            for c, buc in enumerate(bur):
                if buc == -1:
                    pic[r, c] = 0
                if buc == 1:
                    pic[r, c] = 1
        final += pic
    for r, bur in enumerate(final):
        for c, buc in enumerate(bur):
            if buc > 0:
                final[r, c] = 255
    return final
