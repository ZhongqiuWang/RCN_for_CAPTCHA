import os
import numpy as np
import cv2 as cv


def get_training_data_iters(data_dir, train_size, seed, target_dir):
    """
    读取训练数据集
    :param data_dir: str 文件目录
    :param train_size: int
    :param seed:
    :param target_dir:
    :return: train_set/test_set  list  每个元素为一个二元tuple, tuple中分别是表示图片的200*200np.ndarray、表示标签的str
                                        list长度为train_size/test_size
    """
    if not os.path.isdir(data_dir):    # 判断data_dir是否是一个文件目录
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class, target):    # 函数名前有下划线, 不能被外部import
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):    # 从文件夹'3'到文件夹'Z'遍历
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue

            samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class, replace=False)
            # 随机选择num_per_class个, 无放回抽样
            for fname in samples:
                filepath = os.path.join(cat_path, fname)

                # resize和pad的值需要修改
                # Resize and pad the images to (200, 200)
                image_arr = cv.resize(cv.imread(filepath, 0), (112, 112))
                img = np.pad(image_arr, pad_width=tuple([(p, p) for p in (44, 44)]), mode='constant', constant_values=0)
                # pad_width=((44,44),(44,44)), 填补指定值0, 补0后shape为(200, 200)
                loaded_data.append((img, category))

                # 可视化训练样本
                new_name = category + '_' + fname
                target_file = os.path.join(target, new_name)
                cv.imwrite(target_file, img)
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
    # os.path.join() Windows下采用'\\'连接, 而输入地址'/'连接, 注意测试是否可用
    train_set_dir = os.path.join(target_dir, 'train_set')
    _remove_all(train_set_dir)
    train_set = _load_data(os.path.join(data_dir, 'training'), train_size // 28, train_set_dir)
    return train_set
