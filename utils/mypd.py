"""
@File : mypd.py
@Author : 杨与桦
@Time : 2023/12/12 21:59
"""
import pandas as pd
from typing import Union
from pandas.api.types import CategoricalDtype


def snsdt_trans(df, typ: list, retain: Union[list, None] = None):
    tmp = df.unstack().reset_index()
    trans_df = tmp[tmp['level_0'].isin(typ)]
    if not(retain is None):
        retain = df[retain]
        trans_df = pd.merge(trans_df, retain, left_on='level_1', right_on=retain.index)
    trans_df = trans_df.rename(columns={'level_0': 'type', 0: 'value'}).reset_index(drop=True)
    trans_df.drop(columns='level_1', inplace=True)
    return trans_df


def order(df: pd.DataFrame, by: list, order_dict: dict, ascending=True):
    for k in order_dict.keys():
        order_tmp = CategoricalDtype(order_dict[k], ordered=True)  # 实例化有序序列
        df[k] = df[k].astype(order_tmp)  # 将该列转化为有序数据类型
    ordered_df = df.sort_values(by, ascending=ascending).reset_index(drop=True)  # 排序
    return ordered_df


def poi2od(poi_data: Union[pd.DataFrame, pd.Series], edge_col: Union[list, None] = None, o_fix='_o', d_fix='_d'):
    """
将连续的点数据转化为具有OD结构的数据
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


def incond(col: pd.Series, interval: str):
    vals = interval[1: -1].split(',')
    a, b = eval(vals[0]), eval(vals[1])
    a_cond = {'[': col >= a, '(': col > a}
    b_cond = {']': col <= b, ')': col < b}
    condition = (a_cond[interval[0]]) & (b_cond[interval[-1]])
    return condition


def win_clean(data: pd.DataFrame, win: int, subset: list):
    times = (len(data)//win) + 1  # 去重次数
    remain = []
    for i in range(times):
        tmp = data.iloc[i*win: (i+1)*win, :]  # 获取时间窗口内的数据
        tmp: pd.DataFrame
        tmp = tmp.drop_duplicates(subset=subset)  # 去重
        remain.append(tmp)
    cleaned_data = pd.concat(remain)
    return cleaned_data
