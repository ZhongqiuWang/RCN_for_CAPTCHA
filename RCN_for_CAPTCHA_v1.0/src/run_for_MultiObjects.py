import os
import time

from main_process import run_train_experiment, run_parse

"""
默认参数default_parameter
"""
default_parameter = {
    'run_params': 3,  # 运行参数  1：仅训练；2：仅测试；3：训练&测试
    'train_data_dir': '../Dataset/botdetect',
    'test_data_dir': '../Dataset/botdetect/test_set/ancient_mosaic',
    'train_size': 672,  # 28x24
    'test_size': 1,
    'pool_shape': (21, 21),
    'perturb_factor': 3,
    'parallel': True,
    'seed': 1,  # 随机种子 np.random.randint(0, 100)
    'target_data_dir': '../tmp/target_data',  # 将训练图片与测试图片取出放至target_data，使用数据可视
    'model_dir': '../Model/rcn_model_for_MultiObjects_0214test',
    'is_show_tmp': False
}


if __name__ == '__main__':
    print('主程序开始')
    start = time.perf_counter()

    # 训练
    if default_parameter['run_params'] == 1 or default_parameter['run_params'] == 3:
        print('Starting training ......')
        start1 = time.perf_counter()
        run_train_experiment(train_data_dir=default_parameter['train_data_dir'],
                             train_size=default_parameter['train_size'],
                             perturb_factor=default_parameter['perturb_factor'],
                             parallel=default_parameter['parallel'],
                             seed=default_parameter['seed'],
                             target_data_dir=default_parameter['target_data_dir'],
                             model_dir=default_parameter['model_dir'],
                             is_show_tmp=default_parameter['is_show_tmp'])
        end1 = time.perf_counter()
        print('The time of train on %d images: %f Seconds' % (default_parameter['train_size'], (end1 - start1)))

    # 测试
    if default_parameter['run_params'] == 2 or default_parameter['run_params'] == 3:
        print('Starting testing ......')
        start2 = time.perf_counter()
        run_parse(test_data_dir=default_parameter['test_data_dir'],
                  test_size=default_parameter['test_size'],
                  pool_shape=default_parameter['pool_shape'],
                  seed=default_parameter['seed'],
                  target_data_dir=default_parameter['target_data_dir'],
                  model_dir=default_parameter['model_dir'],
                  is_show_tmp=default_parameter['is_show_tmp'])
        end2 = time.perf_counter()
        print('The time of test on %d images: %f Seconds' % (default_parameter['test_size'], (end2 - start2)))

    end = time.perf_counter()
    print('The time of the program: %s Seconds' % (end - start))
