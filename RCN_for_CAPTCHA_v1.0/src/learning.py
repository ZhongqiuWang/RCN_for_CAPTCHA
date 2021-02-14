import numpy as np
import cv2 as cv
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance, cKDTree
from collections import namedtuple

from preprocess import Preproc


ModelFactors = namedtuple('ModelFactors', ['frcs', 'edge_factors', 'graph'])


def train_image(img, perturb_factor, is_show_tmp=False):
    """
    在img上训练模型
    :param img: 2D numpy.ndarray  The training image. 一次学习一张图? 是的
    :param perturb_factor: How much two points are allowed to vary on average given the distance between them.
    :param is_show_tmp: 是否显示中间过程结果，只适合用来看单张训练图片的
    :return: frcs: numpy.ndarray of numpy.int
                   Nx3 array of (feature idx, row, column), where each row represents a single pool center
             edge_factors: numpy.ndarray of numpy.int
                           Nx3 array of (source pool index, target pool index, perturb_radius), where each
                                                   row is a pairwise constraints on a pair of pool choices.
                           池???
             graph: networkx.Graph
                    An undirected graph whose edges describe the pairwise constraints between the pool centers.
                    The tightness of the constraint is in the 'perturb_radius' edge attribute.
    """
    # Pre-processing layer (cf. Sec 4.2.1)
    preproc_layer = Preproc()
    bu_msg = preproc_layer.fwd_infer(img)    # 每张图变为(16, 200, 200), 二值化为-1, 1
    if is_show_tmp:
        show_bu_msg(bu_msg)
    # Sparsification (cf. Sec 5.1.1)
    frcs = sparsify(bu_msg)
    if is_show_tmp:
        show_frcs(frcs)
    # Lateral learning (cf. 5.2)
    graph, edge_factors = learn_laterals(frcs, bu_msg, perturb_factor=perturb_factor)
    # edge_factors = np.array([(edge_source, edge_target, edge_attrs['perturb_radius'])
    #                          for edge_source, edge_target, edge_attrs in graph.edges_iter(data=True)])
    # 上面这句是我写的注释……用来更容易看清edge_factors的数据类型……

    return ModelFactors(frcs, edge_factors, graph)


def sparsify(bu_msg, suppress_radius=3):
    """
    Make a sparse representation of the edges by greedily selecting features from the
                output of preprocessing layer and suppressing overlapping activations.

    Parameters
    ----------
    bu_msg: 3D numpy.ndarray of float
            The bottom-up messages from the preprocessing layer.
            Shape is (num_feats, rows, cols)
    suppress_radius: int
                     How many pixels in each direction we assume this filter
                     explains when included in the sparsification.

    Returns
    -------
    frcs: see train_image.
    """
    frcs = []
    img = bu_msg.max(0) > 0  # img.shape=(200, 200) 每个元素为bool, z方向上有1的为True
    while True:
        r, c = np.unravel_index(img.argmax(), img.shape)  # img中第1个True的索引
        if not img[r, c]:  # 如果img中没有True了, 结束
            break
        # 这个 True 在bu_msg中的 (通道数, 行, 列)
        frcs.append((bu_msg[:, r, c].argmax(), r, c))
        # 抑制以(r, c)为中心7*7的范围
        img[r - suppress_radius:r + suppress_radius + 1, c - suppress_radius:c + suppress_radius + 1] = False
    return np.array(frcs)


def learn_laterals(frcs, bu_msg, perturb_factor, use_adjaceny_graph=False):
    """
    Given the sparse representation of each training example,
    learn perturbation laterals. See train_image for parameters and returns.
    """
    if use_adjaceny_graph:  # 使用邻接图?干啥的?
        graph = make_adjacency_graph(frcs, bu_msg)
        graph = adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)
    else:
        graph = nx.Graph()  # 创建一个空的无向图
        graph.add_nodes_from(range(frcs.shape[0]))  # 图的节点个数等于特征个数N, 节点编号为0 - N-1

    graph = add_underconstraint_edges(frcs, graph, perturb_factor=perturb_factor)
    graph = adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)
    # show_graph(graph, frcs)    # 显示带横向连接关系的图

    edge_factors = np.array([(edge_source, edge_target, edge_attrs['perturb_radius'])
                             for edge_source, edge_target, edge_attrs in graph.edges_iter(data=True)])
    return graph, edge_factors


def show_bu_msg(bu_msg):
    final = np.zeros((200, 200), dtype=np.uint8)
    for i in range(16):
        pic = np.zeros((200, 200), dtype=np.uint8)
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
    dir1 = '../tmp/learning/learning_bu_msg.bmp'
    cv.imwrite(dir1, final)


def show_frcs(frcs):
    img = np.zeros((16, 200, 200), dtype=np.uint8)
    fin = np.zeros((200, 200), dtype=np.uint8)
    for frc in frcs:
        img[frc[0], frc[1], frc[2]] = 255
        fin[frc[1], frc[2]] = 255
    for i in range(16):
        dir1 = '../tmp/learning/'+str(i)+'.bmp'
        cv.imwrite(dir1, img[i])
    dir1 = '../tmp/learning/all_frcs.bmp'
    cv.imwrite(dir1, fin)


def show_graph(graph, frcs):
    coordinates = frcs[:, 1:]
    vnode = np.array(coordinates)
    npos = dict(zip(graph.nodes(), vnode))  # 获取节点与坐标之间的映射关系，用字典表示
    nlabels = dict(zip(graph.nodes(), graph.nodes()))  # 标志字典，构建节点与标识点之间的关系
    nx.draw_networkx_nodes(graph, npos, node_size=70, node_color="#6CB6FF")  # 绘制节点
    # nx.draw_networkx_edges(graph, npos, graph.edges())  # 绘制边
    nx.draw_networkx_labels(graph, npos, nlabels)  # 标签
    plt.show()


def make_adjacency_graph(frcs, bu_msg, max_dist=3):    # 这种方法先不管
    """Make a graph based on contour adjacency."""
    preproc_pos = np.transpose(np.nonzero(bu_msg > 0))[:, 1:]
    preproc_tree = cKDTree(preproc_pos)
    # Assign each preproc to the closest F1
    f1_bus_tree = cKDTree(frcs[:, 1:])
    _, preproc_to_f1 = f1_bus_tree.query(preproc_pos, k=1)
    # Add edges
    preproc_pairs = np.array(list(preproc_tree.query_pairs(r=max_dist, p=1)))
    f1_edges = np.array(list({(x, y) for x, y in preproc_to_f1[preproc_pairs] if x != y}))

    graph = nx.Graph()
    graph.add_nodes_from(range(frcs.shape[0]))
    graph.add_edges_from(f1_edges)
    return graph


def add_underconstraint_edges(frcs, graph, perturb_factor=2., max_cxn_length=100, tolerance=4):
    """
    Examines all pairs of variables and greedily adds pairwise constraints
    until the pool flexibility matches the desired amount of flexibility specified by
    perturb_factor and tolerance.

    Parameters
    ----------
    frcs: numpy.ndarray of numpy.int
          Nx3 array of (feature idx, row, column), where each row represents a single pool center.
    graph:
    perturb_factor : float
                     How much two points are allowed to vary on average given the distance between them.
    max_cxn_length : int
                     The maximum radius to consider adding laterals.
    tolerance : float
                How much relative error to tolerate in how much two points vary relative to each other.

    Returns
    -------
    graph: see train_image.
    """
    graph = graph.copy()  # 为什么要copy一下?
    f1_bus_tree = cKDTree(frcs[:, 1:])  # 用于快速最近邻查找

    close_pairs = np.array(list(f1_bus_tree.query_pairs(r=max_cxn_length)))
    # 得到所有距离小于等于100的点对, 差不多是全部点对
    dists = [distance.euclidean(frcs[x, 1:], frcs[y, 1:]) for x, y in close_pairs]  # close_pairs中每个点对间的距离

    for close_pairs_idx in np.argsort(dists):  # dists中从小到大排序后的原索引
        source, target = close_pairs[close_pairs_idx]
        dist = dists[close_pairs_idx]

        try:
            perturb_dist = nx.shortest_path_length(graph, source, target, 'perturb_radius')
        except nx.NetworkXNoPath:
            perturb_dist = np.inf

        target_perturb_dist = dist / float(perturb_factor)
        actual_perturb_dist = max(0, np.ceil(target_perturb_dist))  # target_perturb_dist向上取整
        if perturb_dist >= target_perturb_dist * tolerance:
            graph.add_edge(source, target, perturb_radius=int(actual_perturb_dist))  # perturb_radius边的权重

    return graph


def adjust_edge_perturb_radii(frcs, graph, perturb_factor=2):
    """
    Returns a new graph where the 'perturb_radius' has been adjusted to account for
    rounding errors. See train_image for parameters and returns.
    调整perturb_radius边的权重, 由于舍入误差
    为什么要这样调整而不在添加边的时候直接进行四舍五入? 为什么下一次迭代要算上上一次的误差?
    如果没有这个调整效果会差多少?
    """
    graph = graph.copy()

    total_rounding_error = 0
    # 直到源结点遍历完所有它能到达的边才停止, n1 n2为每一访问的边的两端的节点
    for n1, n2 in nx.edge_dfs(graph):
        desired_radius = distance.euclidean(frcs[n1, 1:], frcs[n2, 1:]) / perturb_factor

        upper = int(np.ceil(desired_radius))
        lower = int(np.floor(desired_radius))
        round_up_error = total_rounding_error + upper - desired_radius
        round_down_error = total_rounding_error + lower - desired_radius
        if abs(round_up_error) < abs(round_down_error):
            graph.edge[n1][n2]['perturb_radius'] = upper
            total_rounding_error = round_up_error
        else:
            graph.edge[n1][n2]['perturb_radius'] = lower
            total_rounding_error = round_down_error

    return graph
