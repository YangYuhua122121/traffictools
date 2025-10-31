"""
@File : mypd.py
@Author : 杨与桦
@Time : 2023/12/12 21:59
"""
import pandas as pd
from typing import Union
from pandas.api.types import CategoricalDtype

__all__ = ['col2val', 'diy_order', 'incond', 'unit_split']


def col2val(df, typ: list, retain: Union[list, None] = None):
    tmp = df.unstack().reset_index()
    trans_df = tmp[tmp['level_0'].isin(typ)]
    if not(retain is None):
        retain = df[retain]
        trans_df = pd.merge(trans_df, retain, left_on='level_1', right_on=retain.index)
    trans_df = trans_df.rename(columns={'level_0': 'type', 0: 'value'}).reset_index(drop=True)
    trans_df.drop(columns='level_1', inplace=True)
    return trans_df


def diy_order(df: pd.DataFrame, by: list, order_dict: dict, ascending=True):
    for k in order_dict.keys():
        order_tmp = CategoricalDtype(order_dict[k], ordered=True)  # 实例化有序序列
        df[k] = df[k].astype(order_tmp)  # 将该列转化为有序数据类型
    ordered_df = df.sort_values(by, ascending=ascending).reset_index(drop=True)  # 排序
    return ordered_df


def incond(col: pd.Series, interval: str):
    vals = interval[1: -1].split(',')
    a, b = eval(vals[0]), eval(vals[1])
    a_cond = {'[': col >= a, '(': col > a}
    b_cond = {']': col <= b, ')': col < b}
    condition = (a_cond[interval[0]]) & (b_cond[interval[-1]])
    return condition


def unit_split(x1_col=None, x2_col=None, poi: list = None, edge: list = None, left: list = None, right: list = None,
               na_fill=0):
    lst = []

    # 获取填充值
    if type(na_fill) != list:
        na_fill = [na_fill] * len(edge)

    # 解析edge数据
    for i, fill in zip(edge, na_fill):
        i: pd.DataFrame
        i = i.copy()
        # 将起始里程划分
        s = i.drop(columns=[x2_col])
        d = i[[x2_col]].rename(columns={x2_col: x1_col})
        s[x1_col], d[x1_col] = s[x1_col].astype(int), d[x1_col].astype(int)
        dt = pd.concat([s, d])
        dt = dt.fillna(fill)
        lst.append(dt)

    # 解析right数据
    for i in right:
        i = i.copy()
        i[x2_col] = i[x2_col].shift()  # 更改为起点里程
        i[x2_col] = i[x2_col].fillna(0)  # 将第一条数据的起点里程更改为0
        i[x2_col] = i[x2_col].astype(int)
        i = i.rename(columns={x2_col: x1_col})
        lst.append(i)

    # 解析left与poi数据
    for i in left + poi:
        i = i.copy()
        i[x1_col] = i[x1_col].astype(int)
        lst.append(i)

    # 数据融合
    tmp = lst[0]
    for i in lst[1:]:
        tmp = pd.merge(tmp, i, how='outer', on=x1_col)

    # 数据补缺
    # edge数据补缺
    for i, fill in zip(edge, na_fill):
        i: pd.DataFrame
        tmp_col = i.drop(columns=[x1_col, x2_col]).columns
        tmp[tmp_col] = tmp[tmp_col].ffill()
        tmp[tmp_col] = tmp[tmp_col].fillna(fill)
    # left数据补缺
    for i in left:
        tmp_col = i.drop(columns=x1_col).columns
        tmp[tmp_col] = tmp[tmp_col].ffill()
    # right数据补缺
    for i in right:
        tmp_col = i.drop(columns=x2_col).columns
        tmp[tmp_col] = tmp[tmp_col].ffill()

    tmp = tmp.rename(columns={x1_col: '起点里程'})
    tmp['终点里程'] = tmp['起点里程'].shift(-1)
    tmp = tmp.iloc[:-1, :]
    tmp['终点里程'] = tmp['终点里程'].astype(int)
    tmp['单元长度'] = tmp['终点里程'] - tmp['起点里程']
    initial_col = ['起点里程', '终点里程', '单元长度']
    tmp = tmp[initial_col + list(tmp.drop(columns=initial_col).columns)]
    return tmp

def one2one_check(df: pd.DataFrame, col1: str|int, col2: str|int) -> bool:
    """
    检查dataframe中的两列是否为一一对应关系
    :param df: 数据
    :param col1: 字段1
    :param col2: 字段2
    :return: 是否为一一对应关系
    """
    # 检查每组col1对应唯一的col2，且每组col2对应唯一的col1
    grouped1 = df.groupby(col1)[col2].nunique()
    grouped2 = df.groupby(col2)[col1].nunique()

    # 如果所有组的唯一值数量都是1，则说明一一对应
    is_one_to_one = (grouped1 == 1).all() and (grouped2 == 1).all()

    return is_one_to_one

# def win_clean(data: pd.DataFrame, win: int, subset: list):
#     times = (len(data)//win) + 1  # 去重次数
#     remain = []
#     for i in range(times):
#         tmp = data.iloc[i*win: (i+1)*win, :]  # 获取时间窗口内的数据
#         tmp: pd.DataFrame
#         tmp = tmp.drop_duplicates(subset=subset)  # 去重
#         remain.append(tmp)
#     cleaned_data = pd.concat(remain)
#     return cleaned_data
