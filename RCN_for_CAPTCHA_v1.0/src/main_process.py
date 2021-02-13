from multiprocessing import Pool
from functools import partial

from src.get_training_data_for_MultiObject import get_training_data_iters
from src.learning import train_image
import src.save_model as save_model
from src.get_test_data import get_captcha_data_iters
from src.parse_img import ParseSingleCaptcha, parse_single_captcha


def run_train_experiment(train_data_dir, train_size, perturb_factor, parallel, seed, target_data_dir,
                         model_dir, is_show_tmp=False):
    num_workers = None if parallel else 1    # 如果参数parallel为True, num_workers为None, 否则为1
    pool = Pool(num_workers)    # 使用num_workers个工作进程数, 为None时使用个数等于本地CPU数

    # 读入训练数据
    train_data = get_training_data_iters(train_data_dir, train_size, seed, target_data_dir)

    # 训练
    train_partial = partial(train_image, perturb_factor=perturb_factor, is_show_tmp=is_show_tmp)
    # 固定perturb_factor的值, 得到函数train_partial
    train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(999999)
    # train_results: list 每个元素为一个namedtuple——ModelFactors, 包含['frcs', 'edge_factors', 'graph']三个属性

    all_model_factors = zip(*train_results)    # 解压, 3*train_size

    all_model_factors = list(all_model_factors)  # 好像得这样处理一下
    # len(all_model_factors) = 3

    # 保存训练好的模型参数到本地
    save_model.save_model_to_local(all_model_factors, model_dir)


def run_parse(test_data_dir, test_size, pool_shape, seed, target_data_dir, model_dir, is_show_tmp=False):
    # 读入训练好RCN的模型参数
    all_model_factors = save_model.load_model_from_local(model_dir)
    # all_model_factors格式: 长度为3的list, 分别表示['frcs', 'edge_factors', 'graph']
    #     all_model_factors[0]: 长度为672的tuple, 分别表示来自一张训练图片的frcs
    #         all_model_factors[0][0]: 表示第0张训练图片得到的frcs, shape为(52, 3)的np.ndarray

    # 获取测试图片
    test_data = get_captcha_data_iters(test_data_dir, test_size, target_data_dir, seed)
    # test_data是长度为test_size的list, 其中每个元素为一个二元tuple, (img, label)

    # 解析所有测试图片
    true_labels = []
    test_results = []
    parse_right_num = 0
    candidates_right_num = 0
    cnt = 0
    for test_img in test_data:
        cnt += 1
        parse_result = ParseSingleCaptcha(test_img[0], all_model_factors, pool_shape, is_show_tmp)
        true_labels.append(test_img[1].upper())
        print(list(test_img[1].upper()), parse_result.parse_result)
        if list(test_img[1].upper()) == parse_result.parse_result:
            parse_right_num += 1
        candidates_right = judge_candidates(test_img[1].upper(), parse_result.candidates_labels)
        if candidates_right:
            candidates_right_num += 1
        tmp = ("".join(parse_result.parse_result), parse_result.score, parse_result.score_terms, candidates_right)
        test_results.append(tmp)
        print(str(cnt)+'/'+str(test_size), ' parse_right:', parse_right_num, ' candidates_right:', candidates_right_num)

    txt_dir = 'tmp/parsing_results.txt'
    list2txt(true_labels, test_results, txt_dir)
    print('parsing_accuracy =', parse_right_num/test_size)
    print('candidates_accuracy =', candidates_right_num/test_size)


def list2txt(str_list1, test_results, txt_dir):
    with open(txt_dir, 'w') as file_object:
        for i in range(len(str_list1)):
            sp = '    '
            str_w1 = str_list1[i]+sp+test_results[i][0]+sp+str(test_results[i][1])
            str_w2 = sp+str(test_results[i][2])+sp+str(test_results[i][3])+'\n'
            str_w = str_w1+str_w2
            file_object.write(str_w)


def judge_candidates(true_label, candidates_labels):
    """
    评价RCN模型得到的候选者, 只判断candidates_labels是否都包含true_label
    :param true_label: 测试图片真实标签, str
    :param candidates_labels: candidates的识别结果, list, 每个元素是一个字符
    :return:
    """
    true_label_length = len(true_label)
    candidates_num = len(candidates_labels)
    start = 0
    right_num = 0
    for i in range(true_label_length):
        for j in range(start, candidates_num):
            if candidates_labels[j] == true_label[i]:
                right_num += 1
                start = j + 1
                break
    return right_num == true_label_length
