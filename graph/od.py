"""
@File : od.py
@Author : 杨与桦
@Time : 2024/05/20 20:01
"""
import pandas as pd
import numpy as np
from shapely import linestrings
from typing import Union


def poi2od_df(poi_data: Union[pd.DataFrame, pd.Series], edge_col: Union[list, None] = None, o_fix='_o', d_fix='_d'):
    """
将连续的点数据(表格)转化为具有OD结构的数据
    :param poi_data:点数据
    :param edge_col:指定OD边属性对应的列，如['ID', 'state']，可缺省
    :param o_fix:O点端点属性的后缀名
    :param d_fix:D点端点属性的后缀名
    :return:OD结构的数据
    """
    if type(poi_data) == pd.Series:
        poi_data = poi_data.to_frame()
    poi_data = poi_data.copy()

    # 对O列更名
    o_col = [str(i) + o_fix for i in poi_data.columns]
    poi_data = poi_data.rename(columns=dict(zip(poi_data.columns, o_col)))

    # 平移以构造D列
    for i in poi_data.columns:
        i = str(i)
        poi_data[i.rstrip(o_fix) + d_fix] = poi_data[i].shift(-1)

    # 删除不对应的列并整合，当且仅当边属性一致的数据被保留
    if isinstance(edge_col, list):
        for i in edge_col:
            i = str(i)
            poi_data = poi_data[poi_data[i + o_fix] == poi_data[i + d_fix]]

        od_data = poi_data.copy()
        # 删除边属性的‘d’后缀
        od_data.drop(columns=[str(i) + d_fix for i in edge_col], inplace=True)
        # 将'o'后缀改为无后缀
        tmp = [str(i) + o_fix for i in edge_col]
        od_data.rename(columns=dict(zip(tmp, edge_col)), inplace=True)
    else:
        od_data = poi_data[:-1].copy()

    return od_data


def poi2od_lst(path_pois: list):
    """
将路径点转化为路径边（输入为列表）
    :param path_pois:
    :return:
    """
    path_ods = []
    for i in range(len(path_pois) - 1):
        path_ods.append(path_pois[i: i + 2])
    return path_ods


def xy2od(oxy: pd.DataFrame, dxy: pd.DataFrame):
    """
借助OD点的xy坐标高效生成OD对的线矢量(LineString)
    :param oxy:
    :param dxy:
    :return:
    """
    xy = pd.concat([oxy, dxy], axis=1)
    xy = xy.values.reshape(-1, 2)
    indices = np.repeat(range(int(len(xy) / 2)), 2)
    return linestrings(xy, indices=indices)


def od_undirect(df: pd.DataFrame, link: list[any, any]) -> pd.DataFrame:
    def historic(n):
        def prime(ii, primes_s):
            for prime_e in primes_s:
                if not (ii == prime_e or ii % prime_e):
                    return False
            primes_s.add(ii)
            return ii
        primes = {2}
        i, pr = 2, 0
        while True:
            if prime(i, primes):
                pr += 1
                if pr == n:
                    return primes
            i += 1
    p1 = list(df[link[0]].unique())
    p2 = list(df[link[1]].unique())
    p = list(set(p1+p2))
    p1_mark = pd.DataFrame({link[0]: p, 'mark1': list(historic(len(p)))})
    p2_mark = pd.DataFrame({link[1]: p, 'mark2': list(historic(len(p)))})
    df = pd.merge(df, p1_mark, on=link[0])
    df = pd.merge(df, p2_mark, on=link[1])
    df['mark'] = df['mark1']*df['mark2']
    new_link = df.drop_duplicates(subset='mark')[link+['mark']]
    df = pd.merge(df, new_link, on='mark')
    df = df.drop(columns=[str(link[0])+'_x', str(link[1])+'_x', 'mark1', 'mark2', 'mark'])
    df.rename(columns={str(link[0])+'_y': link[0], str(link[1])+'_y': link[1]}, inplace=True)
    return df


def od_undirect2(df: pd.DataFrame, link: list[any, any]) -> pd.DataFrame:
    df1 = df[[link[0]]+[link[1]]+list(df.columns.drop(link))].copy()
    df1['mark1'] = range(len(df))
    df2 = df[[link[1]]+[link[0]]+list(df.columns.drop(link))].copy()
    df2.rename(columns={link[0]: link[1], link[1]: link[0]}, inplace=True)
    df2['mark1'] = range(len(df))
    df = pd.concat([df1, df2])
    lst = []
    for i in df.groupby(link):
        i[1]['mark1'] = i[1]['mark1'].min()
        i[1]['mark2'] = range(len(i[1]))
        lst.append(i[1])
    res = pd.concat(lst)
    undi_df = res.drop_duplicates(subset=['mark1', 'mark2']).drop(columns=['mark1', 'mark2'])
    return undi_df
