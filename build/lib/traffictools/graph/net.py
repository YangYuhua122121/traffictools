"""
@File : net.py
@Author : 杨与桦
@Time : 2023/11/29 18:05
"""
import pandas as pd
import networkx as nx


def df2net(df: pd.DataFrame, n1, n2, other=None):
    g = nx.Graph()
    if other is None:  # 无属性
        g.add_edges_from(list(zip(df[n1], df[n2])))
    else:
        for i in df.index:
            tmp = df.loc[i]
            other_dict = {}
            for k in other:
                other_dict[k] = tmp[k]
            g.add_edge(tmp[n1], tmp[n2], **other_dict)
    return g


def net2df(graph: nx.Graph, n1='node1', n2='node2'):
    raw = pd.DataFrame(graph.edges(data=True))
    df = pd.concat([raw[[0, 1]], pd.DataFrame(raw[2].tolist())], axis=1)
    df.rename(columns={0: n1, 1: n2}, inplace=True)
    return df


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
