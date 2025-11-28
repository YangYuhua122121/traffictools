"""
@File : net.py
@Author : 杨与桦
@Time : 2023/11/29 18:05
"""
import pandas as pd
import networkx as nx
import numpy as np
from typing import Union, List, Tuple
from od import poi2od_lst


def df2net(df: pd.DataFrame, n1, n2, is_dg: bool = True, attr: Union[None, list] = None):
    if is_dg:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    if attr is None:  # 无属性
        g.add_edges_from(list(zip(df[n1], df[n2])))
    else:  # 有属性
        for i, tmp in df.iterrows():
            other_dict = {}
            for k in attr:
                other_dict[k] = tmp[k]
            g.add_edge(tmp[n1], tmp[n2], **other_dict)
    return g


def net2df(graph: nx.Graph, n1='O', n2='D', p='name', mode: str = 'edge'):
    if mode == 'edge':
        raw = pd.DataFrame(graph.edges(data=True))
        df = pd.concat([raw[[0, 1]], pd.DataFrame(raw[2].tolist())], axis=1)
        df.rename(columns={0: n1, 1: n2}, inplace=True)
    elif mode == 'node':
        raw = pd.DataFrame(graph.nodes(data=True))
        df = pd.concat([raw[[0]], pd.DataFrame(raw[1].tolist())], axis=1)
        df.rename(columns={0: p}, inplace=True)
    else:
        raise ValueError(f'mode={mode}不是可选项（edge 或 node）')
    return df


# 基于邻居的布局算法，用于单源单汇运输网络
def neighbor_layout(graph: nx.DiGraph, level_dx=20, inner_dx=5, inner_dy=10, dy=10):
    """
用于单源单汇运输网络（for nx.DiGraph）的布局算法，返回值与nx内置布局算法一致
该函数返回一个字典:{'node': [x, y], ...}， 可直接用于nx.draw(pos=pos)
    :param graph: networkx的有向图
    :param level_dx: 层间节点的水平间距参数
    :param inner_dx: 层内节点的水平间距参数
    :param inner_dy: 层内节点的垂直间距参数
    :param dy:  图的垂直高度参数
    :return:节点坐标字典
    """
    # 检索源和汇
    begin, end = spoi_search(graph)[0][0], spoi_search(graph)[1][0]

    this_group = [begin]  # 本次搜索的主对象
    pos_dict = {begin: [0, 0]}
    x, y, sign_x, sign_y = level_dx, 0, -1, -1
    n = 0

    while len(pos_dict) != len(graph.nodes):  # 循环至全部节点的坐标均被定义
        next_group = []  # 重置邻居队列

        # 获取全部邻居节点，并添加至队列中
        for i in this_group:
            tmp_dict = {}
            next_group += list(graph.neighbors(i))

            sign_y *= -1  # 同组内偏向一致
            for j in list(graph.neighbors(i)):
                sign_x *= -1
                x += sign_x * inner_dx  # 波动
                y += inner_dy
                tmp_dict[j] = [x, sign_y * y]

            # 更新原字典，不覆盖已有值
            tmp_dict.update(pos_dict)
            pos_dict = tmp_dict

        # 本次主对象的邻居赋值完毕后，更新搜索的主对象，并将x、y转换至下一层
        this_group = next_group
        y = 0
        x += level_dx
        n += 1

    # 修正坐标：标准化（z-score）每层的纵坐标、定位结点
    xy = np.array(list(pos_dict.values()))
    for i in range(1, 1 + n):
        idx = np.argwhere((xy[:, 0] >= level_dx * i - inner_dx) & (xy[:, 0] <= level_dx * i + inner_dx))
        if xy[idx][:, :, 1].std() == 0:
            std = 1
        else:
            std = xy[idx][:, :, 1].std()
        xy[idx, 1] = dy * (xy[idx][:, :, 1] - xy[idx][:, :, 1].mean()) / std
    for i, j in zip(pos_dict.keys(), xy):
        pos_dict[i] = list(j)

    pos_dict[end] = [(n + 1) * level_dx, 0]

    return pos_dict


def spoi_search(graph: nx.DiGraph):  # 搜索源和汇
    source = []
    sink = []
    for i in list(graph.nodes):
        if graph.in_degree(i) == 0:
            source.append(i)
        if graph.out_degree(i) == 0:
            sink.append(i)
    return source, sink


def poi2e(path_pois, g, label='name', mode='label'):
    edge_lst2 = []  # 获取独立的边的顶点列表[(1, 2), (2, 4)]
    for i in range(len(path_pois) - 1):
        edge_lst2.append(tuple(path_pois[i: i + 2]))  # 以win=2的滑动窗口获取独立顶点边
    edge_lst = []  # 输出
    if mode == 'label':
        for i in edge_lst2:  # 迭代每条边
            edge_name = g.get_edge_data(*i)['weight'][label]  # 获取该边的属性
            edge_lst.append(edge_name)
    elif mode == 'pp':
        edge_lst = edge_lst2
    return edge_lst


# 坐标规则化
def pos_regularize(pos_dict: dict, dx=20, dy=5):
    xy = np.array(list(pos_dict.values()))
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    x, y = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
    y = dy * (y // dy)
    xy = np.concatenate([x.ravel().reshape(-1, 1), y.ravel().reshape(-1, 1)], axis=1)  # 2xn
    new_pos = {}
    for i in pos_dict.keys():
        xyi = np.array(pos_dict[i])
        dist = (np.abs(xyi - xy)).sum(axis=1)  # 该点与网格点的距离
        min_idx = dist.argmin()
        new_pos[i] = list(xy[min_idx, :])  # 将该点坐标赋值为最近网格点
        xy = np.delete(xy, min_idx, 0)
    return new_pos


# 坐标重排序
def pos_resort(pos_dict: dict, before, after):
    if set(before) != set(after):
        raise ValueError('请确保before与after列表中出现过的节点是相同的')
    elif len(before) != len(after):
        raise ValueError('请确保before与after列表中的节点数量相同')
    else:
        pos_dict = pos_dict.copy()
        refer_dict = pos_dict.copy()  # 参考组
        for i, j in zip(before, after):
            pos_dict[i] = refer_dict[j]
    return pos_dict

class NetGenerator:
    def __init__(self):
        """
        生成各类路网模型的拓扑图边列表
        """
    def highway(
            self,
            all_order: list, on: list, off: list, service: List[tuple],
    )-> Tuple[list, dict, list]:
        """
        高速公路线性模型，支持主线、出入匝道及服务区节点的输入
        :param all_order: 线性模型的节点顺序列表
        :param on: 入口匝道的节点顺序列表
        :param off: 出口匝道的节点顺序列表
        :param service: 服务区入出口节点元组列表（先入后出）。如[(3, 4), (8, 9)]表示两个服务区的入出口
        :return: 一个三元组(拓扑图的边列表，自动生成节点坐标，主线节点)
        """
        all_order = list(map(str, all_order))
        on = list(map(str, on))
        off = list(map(str, off))
        service2 = list(map(str, [i for ii in service for i in ii]))  # 服务区节点一维列表
        main_station = [i for i in all_order if (i not in on) and (i not in off) and (i not in service2)]  # 主线门架

        edge_lst = poi2od_lst(all_order)
        pos_dct = {all_order[idx]: [idx * 10, 0] for idx in range(len(all_order))}

        # 处理上匝道
        for s in on:
            edge_lst.append([s + '.1', s])
            x = pos_dct[s][0]
            pos_dct.update({s + '.1': [x, 5]})

        # 处理下匝道
        for s in off:
            edge_lst.append([s, s + '.0'])
            x = pos_dct[s][0]
            pos_dct.update({s + '.0': [x, -5]})

        # 处理服务区
        for ses in service:
            o = str(ses[0])
            d = str(ses[1])
            edge_lst.append([o, o + d])
            edge_lst.append([o + d, d])
            x = pos_dct[o][0]
            pos_dct.update({o + d: [x, -3]})

        return edge_lst, pos_dct, main_station
