"""
@File : detect.py
@Author : 杨与桦
@Time : 2024/03/29 18:25
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from traffictools.graph import od
from traffictools.utils import mypd


class Coil:
    def __init__(self, data: pd.DataFrame, col_dict: Union[None, dict] = None,
                 pos_dict: Union[None, dict] = None):
        """
线圈数据容器
        :param data: 线圈数据
        :param col_dict: 指定关键参数列名。默认值条件下，相关参数列名为['id', 't', 'q', 'v', 'o']
        :param pos_dict: 指定线圈里程位置（m）
        """
        self.__data = data
        self.__pos_dict = pos_dict  # 线圈里程坐标(m)

        # 修改关键参数的列名映射，使用列名时，格式为：col_dict['x']
        self.col_dict = {'id': 'id', 't': 't', 'q': 'q', 'v': 'v', 'o': 'o'}
        if isinstance(col_dict, dict):
            for key in col_dict.keys():
                self.col_dict[key] = col_dict[key]

        # 采样间隔
        sub = list(data.drop_duplicates(subset=self.col_dict['t']).
                   sort_values(by=self.col_dict['t'])[:2][self.col_dict['t']])
        self.T = (sub[1] - sub[0]).total_seconds()

    def travel_time_cal(self, method='instant', detail=False, max_iter: int = 5):
        """
行程时间计算
        :param method:指定计算模型，含 'instant'：瞬时速度模型；'slice'：时间切片模型；'dynamic'：动态时间切片；'linear'：线性模型
        :param detail:是否返回分段行程时间
        :param max_iter:最大迭代次数（仅动态时间切片）
        :return:
        """
        # 基于时间和线圈编号排序数据
        order_list = list(pd.Series(self.__pos_dict).sort_values().index)
        ordered_data = mypd.diy_order(self.__data, by=[self.col_dict['t'], self.col_dict['id']],
                                      order_dict={self.col_dict['id']: order_list})
        v_col = self.col_dict['v']
        id_col = self.col_dict['id']

        # 数据预处理，构造OD数据，求解路段平均速度
        od_data = od.poi2od_df(ordered_data, [self.col_dict['t']])
        od_data['mean_v'] = (od_data[f'{v_col}_o'] + od_data[f'{v_col}_d']) / 2

        # 线圈与路段长度映射
        dst_data = od.poi2od_df(pd.Series(self.__pos_dict).sort_values().reset_index())
        dst_data['interval'] = dst_data['0_d'] - dst_data['0_o']
        dst_data.columns = [f'{id_col}_o', '_', f'{id_col}_d', '_', 'interval']
        dst_data.drop(columns=['_'], inplace=True)

        # 时刻-位置-速度对照表（O点不包含尾部线圈）,并构造时间戳
        tmp = pd.merge(od_data, dst_data)
        tmp['timestamp'] = (tmp[self.col_dict['t']] - tmp[self.col_dict['t']].min()).dt.total_seconds()
        tmp['timestamp'] = tmp['timestamp'].astype(int)
        tmp['travel_time'] = tmp['interval'] / (tmp['mean_v'])  # 单位为秒

        # 尾部线圈的对照表
        tail_data = self.__data[self.__data[self.col_dict['id']] == order_list[-1]].copy()
        tail_data['timestamp'] = (tail_data[self.col_dict['t']] - tmp[self.col_dict['t']].min()).dt.total_seconds()
        tail_data['timestamp'] = tail_data['timestamp'].astype(int)

        # 工具
        # 将路段区间行程时间匹配至路段名
        def od_map(t_value):
            lst = [dst_data] * int((len(t_value) / len(dst_data)))
            x = pd.concat(lst).drop(columns='interval').reset_index(drop=True)  # 构造OD列
            return pd.concat([x, t_value], axis=1)

        # 基于采样间隔取整
        def int_base_t(x):
            y = self.T * ((x / self.T).astype(float).round())
            y = y.astype(int)
            return y

        # 线性模型所用的函数
        def t2x(v0, t, a):
            a = a.copy()
            a[a == 0] = 0.000001
            y = (v0 * (np.exp((a * t).astype(float)) - 1)) / a
            return y

        def x2t(v0, x, a):
            a = a.copy()
            a[a == 0] = 0.000001
            tmp_factor = ((v0 + a * x) / v0).astype(float)
            tmp_factor[tmp_factor < 0] = 0.01  # 避免无解
            y = np.log(tmp_factor) / a
            return y

        # 瞬时速度模型
        if method == 'instant':
            res = tmp[[f'{id_col}_o', f'{id_col}_d', self.col_dict['t'], 'travel_time']]

        # 时间切片模型
        elif method == 'slice':
            # 初始化首个路段区间
            tmp1 = tmp[tmp[f'{id_col}_o'] == order_list[0]][['travel_time', 'timestamp', self.col_dict['t']]]
            tmp1 = tmp1.copy()
            tmp1['cum_t'] = (tmp1['timestamp'] + tmp1['travel_time'])
            tmp1['end_t'] = int_base_t(tmp1['cum_t'])  # 进入下一(取近时刻)
            tmp1['end_t'] = tmp1['end_t'].astype(int)
            tmp1.drop(columns='timestamp', inplace=True)
            tmp1.columns = ['travel_time0', self.col_dict['t'], 'cum_t', 'end_t']
            tmp1 = tmp1[[self.col_dict['t'], 'cum_t', 'end_t', 'travel_time0']]
            # 后续路段区间计算
            for i in range(len(order_list) - 2):
                tmp2 = tmp[tmp[f'{id_col}_o'] == order_list[i + 1]][['travel_time', 'timestamp']]
                tmp1 = pd.merge(tmp1, tmp2, left_on='end_t', right_on='timestamp')
                tmp1['cum_t'] += tmp1['travel_time']
                tmp1['end_t'] = int_base_t(tmp1['cum_t'])
                tmp1['end_t'] = tmp1['end_t'].astype(int)
                tmp1.rename(columns={'travel_time': f'travel_time{i + 1}'}, inplace=True)
                tmp1.drop(columns=['timestamp'], inplace=True)
            tmp1 = tmp1[[self.col_dict['t']] + [f'travel_time{i}' for i in range(len(order_list) - 1)]]
            col = [f'travel_time{i}' for i in range(len(order_list) - 1)]
            tmp2 = mypd.col2val(tmp1, col, [self.col_dict['t']])[[self.col_dict['t'], 'value']]
            res = od_map(tmp2)

        # 动态时间切片模型
        elif method == 'dynamic':
            col = [self.col_dict['t'], f'{v_col}_o', 'interval', 'travel_time', 'timestamp']
            tmp1 = tmp[tmp[f'{id_col}_o'] == order_list[0]][col]
            tmp1 = tmp1.copy()
            tmp1['end_t'] = int_base_t(tmp1['timestamp'] + tmp1['travel_time'])  # 初始值条件下的离开时间（取整）
            tmp1.rename(columns={'timestamp': 'cum_t'}, inplace=True)
            tmp1['Y'] = 1
            tmp1.rename(columns={'travel_time': 'travel_time0'}, inplace=True)
            tmp1 = tmp1[[self.col_dict['t'], 'cum_t', 'Y', 'interval', f'{v_col}_o', 'end_t', 'travel_time0']]

            # 迭代每个线圈
            for i in range(len(order_list) - 1):
                # 获取下游线圈数据
                if order_list[i + 1] == order_list[-1]:  # 若为最后一个线圈
                    tmp2 = tail_data[[v_col, 'timestamp']]
                    tmp2 = tmp2.copy()
                    tmp2.rename(columns={v_col: f'{v_col}_o'}, inplace=True)
                else:
                    tmp2 = tmp[tmp[f'{id_col}_o'] == f'{order_list[i + 1]}'][[f'{v_col}_o', 'timestamp']]

                for _ in range(max_iter):
                    # 若Y含非零值，则迭代
                    if tmp1['Y'].any():
                        tmp1 = pd.merge(tmp1, tmp2, left_on='end_t', right_on='timestamp')  # 匹配下游速度值（离开时间对应的速度）
                        tmp1[f'travel_time{i}'] = ((2 * tmp1['interval']) /
                                                   (tmp1[f'{v_col}_o_x'] + tmp1[f'{v_col}_o_y']))  # 计算新的行程时间
                        # Y:判断原有离开时间是否与新的离开时间一致
                        tmp1['Y'] = tmp1['end_t'] - int_base_t(tmp1['cum_t'] + tmp1[f'travel_time{i}'])
                        tmp1['end_t'] = int_base_t(tmp1['cum_t'] + tmp1[f'travel_time{i}'])  # 更新本路段的离开时间
                        # 此时v_o是离开时间对应的速度，对应于下区段的初始速度
                        tmp1.drop(columns=[f'{v_col}_o_x', 'timestamp'], inplace=True)
                        tmp1.rename(columns={f'{v_col}_o_y': f'{v_col}_o'}, inplace=True)
                    # 若Y全为零（完全一致）
                    else:
                        break

                # 否则，更新数据，进入下一个区段
                tmp1['cum_t'] += tmp1[f'travel_time{i}']  # 更新总时间(不取整)
                tmp1['interval'] = dst_data.loc[i, 'interval']  # 下一路段间距

                tmp1['Y'] = 1  # Y的初始化
                tmp1['end_t'] = int_base_t(tmp1['cum_t'] + tmp1[f'travel_time{i}'])  # 计算下一路段的初始化离开时间

            tmp1.drop(columns=['Y', 'interval', 'end_t', f'{v_col}_o', 'cum_t'], inplace=True)
            tmp2 = (mypd.col2val(tmp1, typ=list(tmp1.columns[1:]), retain=[self.col_dict['t']]).
                    drop(columns='type'))
            res = od_map(tmp2)

        # 线性模型
        elif method == 'linear':
            # 构造路段映射
            coil_pos = pd.Series(self.__pos_dict).reset_index()
            link_map = (pd.merge(dst_data, coil_pos, left_on=f'{id_col}_d', right_on='index').
                        reset_index())
            link_map = link_map.drop(columns=['index', f'{id_col}_d']).rename(columns={'level_0': 'Link', 0: 'end_pos'})
            append = pd.DataFrame(data=[[link_map['Link'].max() + 1, '无穷远处', 9999, 9999]], columns=link_map.columns)
            link_map = pd.concat([link_map, append])

            # 添加无穷远处数据
            append = tmp[tmp[f'{id_col}_o'] == order_list[0]].copy()
            append[f'{id_col}_o'] = '无穷远处'
            append[f'{v_col}_d'] = 0.01
            tmp = pd.concat([tmp, append])

            # 主体
            col = [self.col_dict['t'], f'{v_col}_o', f'{v_col}_d', 'timestamp', 'interval', f'{id_col}_o']
            tmp1 = tmp[tmp[f'{id_col}_o'] == order_list[0]][col]
            tmp1['Timer'] = 0  # 计时器
            tmp1['Pos'] = 0  # 记位器
            tmp1['Link'] = 0  # 路段记录
            tmp1['end_pos'] = 0  # 与下游检测器距离
            n = 0  # 迭代次数

            while not (tmp1[f'{id_col}_o'] == '无穷远处').all():
                tmp1.drop(columns=['interval', 'end_pos', f'{id_col}_o'], inplace=True)
                tmp1 = pd.merge(tmp1, link_map, on='Link')  # 加载下游线圈信息
                tmp1['dv/dx'] = ((tmp1[f'{v_col}_d'] - tmp1[f'{v_col}_o']) /
                                 (tmp1['end_pos'] - tmp1['Pos']))  # 速度变化率，单位为s^-1

                tmp1['dx'] = t2x(tmp1[f'{v_col}_o'], self.T, tmp1['dv/dx'])  # 一个检测周期内行驶的距离
                tmp1['dt'] = x2t(tmp1[f'{v_col}_o'], (tmp1['end_pos'] - tmp1['Pos']), tmp1['dv/dx'])  # 剩余路段内行驶时间

                # 更新位置
                tmp1['last_Pos'] = tmp1['Pos']
                tmp1.loc[tmp1['dt'] > self.T, 'Pos'] += tmp1['dx']
                tmp1.loc[tmp1['dt'] <= self.T, 'Pos'] += tmp1['end_pos'] - tmp1['Pos']  # 驶过剩余路段
                tmp1.loc[tmp1['dt'] <= self.T, 'Link'] += 1  # 进入下一路段

                tmp1.drop(columns=['interval', 'end_pos', f'{id_col}_o'], inplace=True)
                tmp1 = pd.merge(tmp1, link_map, on='Link')  # 及时更新下游线圈信息

                # 周期末速度，更新为上游速度
                tmp1[f'{v_col}_o'] = tmp1[f'{v_col}_o'] + (tmp1['Pos'] - tmp1['last_Pos']) * tmp1['dv/dx']

                # 更新计时器
                tmp1['last_Time'] = tmp1['Timer']
                tmp1.loc[tmp1['dt'] > self.T, 'Timer'] += self.T  # 一个检测周期内未到达下游端的计时器+20
                tmp1.loc[tmp1['dt'] <= self.T, 'Timer'] += tmp1['dt']  # 反之，则+路段行驶时间

                # 记录路段变换时间点
                tmp1.loc[tmp1['dt'] > self.T, f'travel_time{n}'] = 0
                tmp1.loc[tmp1['dt'] <= self.T, f'travel_time{n}'] = tmp1['Timer']

                # 匹配新的下游速度
                tmp1['timestamp'] = int_base_t(tmp1['timestamp'] + tmp1['Timer'] - tmp1['last_Time'])  # 绝对取整时间
                tmp1.drop(columns=f'{v_col}_d', inplace=True)
                tmp1 = pd.merge(tmp1, tmp[['timestamp', f'{v_col}_d', f'{id_col}_o']], on=['timestamp', f'{id_col}_o'])
                n += 1

            travel_time_col = [f'travel_time{i}' for i in range(n)]
            tmp1 = tmp1[[self.col_dict['t']] + travel_time_col]

            # 删除无效的行程时间记录
            tmp2 = mypd.col2val(tmp1, list(tmp1.columns)[1:], retain=[self.col_dict['t']])
            tmp2 = tmp2[tmp2['value'] != 0]  # 删除无效记录（travel_time=0）
            tmp3 = tmp2.drop_duplicates(subset=self.col_dict['t']).copy()
            tmp3['value'] = 0  # 构造初始列，用于后续OD作差
            tmp2 = pd.concat([tmp2, tmp3]).sort_values(by=[self.col_dict['t'], 'value'])
            tmp2 = od.poi2od_df(tmp2[['value', self.col_dict['t']]], [self.col_dict['t']])
            tmp2['travel_time'] = tmp2['value_d'] - tmp2['value_o']
            tmp2 = tmp2.sort_values(by=self.col_dict['t'])
            tmp2 = tmp2.drop(columns=['value_o', 'value_d']).reset_index(drop=True)
            res = od_map(tmp2)

        # 输入了错误的method
        else:
            print(f'方法集（method参数）中不存在{method}！')
            res = -1

        # 行程时间输出格式
        if not ('travel_time' in list(res.columns)):
            res.rename(columns={'value': 'travel_time'}, inplace=True)
        if detail:
            return res
        else:
            return res.groupby(self.col_dict['t'])['travel_time'].sum().reset_index()
