该目录用于保存从数据集中获取的用于训练的图片和用于测试的图片。

用于训练的图片保存在该目录下的`train_set`文件夹中，生成的数据来自`src/get_training_data_for_MultiObject.py`中的`get_training_data_iters()`函数。

用于测试的图片保存在该目录下的`test_set`文件夹中，生成的数据来自`src/get_test_data.py`中的`get_captcha_data_iters()`函数。