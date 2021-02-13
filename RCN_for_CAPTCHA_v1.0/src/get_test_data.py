import os
import shutil
import numpy as np
from cv2 import imread


def get_captcha_data_iters(data_dir, test_size, target_data_dir, seed=5):
    """
    获取测试图片
    :param data_dir: 测试图片文件所在路径
    :param test_size: 测试图片张数
    :param target_data_dir: 将选出的测试图片放到该路径下
    :param seed: 随机种子
    :return: test_set
             长度为test_size的list, 其中每个元素为一个二元tuple, (img, label), label是一个str
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num, target_dir):
        loaded_data = []
        samples = np.random.choice(sorted(os.listdir(image_dir)), num, replace=False)
        tag = 0
        for fname in samples:
            filepath = os.path.join(image_dir, fname)
            img = imread(filepath, 0)    # 以灰度图的形式读取图片
            fname_new = fname[:-4]    # 删去文件名的后缀.png, 剩下的即为label
            loaded_data.append((img, fname_new))    # 要加载的数据

            # 把要加载的数据放入target_dir
            new_name = str(tag)+'_'+fname
            target_file = os.path.join(target_dir, new_name)
            shutil.copy(filepath, target_file)
            tag += 1

        return loaded_data

    def _remove_all(path):
        """
        删除path下所有文件
        """
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    np.random.seed(seed)

    test_data_dir = os.path.join(target_data_dir, 'test_set')
    _remove_all(test_data_dir)

    test_set = _load_data(data_dir, test_size, test_data_dir)

    return test_set
