"""
@File : fourstep.py
@Author : 杨与桦
@Time : 2023/11/28 14:16
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Union
from traffictools.graph import net
from traffictools.utils import opt
import networkx as nx


class Generator:
    def __init__(self, p: pd.DataFrame, a: pd.DataFrame, zone_f='zone', times_f='times'):
        self.p, self.a = p.rename(columns={times_f: 'p_times'}), a.rename(columns={times_f: 'a_times'})
        self.pa = pd.merge(self.p, self.a, on=zone_f)
        self.zone = zone_f
        self.p_model, self.a_model = None, None

    @staticmethod
    def pa_zone_get(trips, o, d, d_home):
        trips.trips.loc[trips.trips[d_home] == 1, 'p_zone'] = trips.trips[d]
        trips.trips.loc[trips.trips[d_home] == 0, 'p_zone'] = trips.trips[o]
        trips.trips.loc[trips.trips[d_home] == 1, 'a_zone'] = trips.trips[o]
        trips.trips.loc[trips.trips[d_home] == 0, 'p_zone'] = trips.trips[d]
        return trips

    def lm_fit(self, mode='p', **kwargs):
        if mode == 'p':
            x = sm.add_constant(self.p[kwargs['x']])
        else:
            x = self.a[kwargs['x']]
        self.a_model = sm.OLS(self.p[kwargs['y']], x).fit()

    def lm_pred(self, mode='p', **kwargs):
        if mode == 'p':
            x = sm.add_constant(self.p[kwargs['x']])
            fp = self.p_model.predict(x)*kwargs['population']
        else:
            x = self.a[kwargs['x']]
            fa = self.a_model.predict(x)

    def trip_balance(self):
        pa = pd.merge(self.p, self.a, on=self.zone)
        pa['a_times'] = (pa['p_times'].sum() / pa['a_times'].sum()) * pa['a_times']
        pa[['p_times', 'a_times']] = pa[['p_times', 'a_times']].astype(int)
        self.pa = pa


class Distributor:
    def __init__(self, pa_mat: Union[np.ndarray, pd.DataFrame], zone_data: pd.DataFrame, mode='growth', **kwargs):
        self.mode = mode
        self.pa_mat = pa_mat  # 现状PA矩阵
        self.zone_data = zone_data
        self.n = 0  # 迭代次数
        if mode == 'growth':  # 增长系数法
            self.gi, self.gj = kwargs['gi'], kwargs['gj']  # 小区增长系数
            self.p2, self.a2 = kwargs['p2'], kwargs['a2']  # 小区目标PA量
        if mode == 'gravity':  # 重力模型法
            self.cost = kwargs['cost']  # 阻抗/代价矩阵

    def solve(self, func, e=0.01, max_iter=100, **kwargs):
        if self.mode == 'growth':
            flag = True
            while flag and (self.n <= max_iter):
                self.n += 1
                g = func(self.cost, **kwargs)  # 增长系数矩阵
                # 获取未来分布
                self.pa_mat = g * self.pa_mat

                # 收敛判定
                self.zone_data[self.gj] = self.zone_data[self.a2] / self.pa_mat.sum(axis=0).reset_index(drop=True)
                self.zone_data[self.gi] = self.zone_data[self.p2] / self.pa_mat.sum(axis=1).reset_index(drop=True)

                gi_ok = ((1 - e <= self.zone_data[self.gi]) & (self.zone_data[self.gi] <= 1 + e)).all()
                gj_ok = ((1 - e <= self.zone_data[self.gj]) & (self.zone_data[self.gj] <= 1 + e)).all()
                flag = not (gi_ok and gj_ok)
        elif self.mode == 'gravity':
            a, p = np.meshgrid(self.zone_data[self.a2], self.zone_data[self.p2])  # P和A的列阵
            f = func()  # 计算摩擦因子矩阵
            f[np.isinf(f)] = 0  # 消除无穷项
            k = np.reciprocal(f @ a[[0], :].T)  # 计算模型系数向量（单约束）
            kk = np.concatenate([k] * len(p), axis=1)  # 获得模型系数矩阵
            self.pa_mat = kk * a * p * f

    def train(self, true_pa, func, e=0.01, max_iter=50, lr=0.1, **params):  # 输入模型的初始化参数
        flag = True
        while flag:
            self.n += 1
            self.solve(func, **params)  # 计算PA分布的估计值
            mean_ec = (np.sum((self.pa_mat * self.cost).values)) / (np.sum(self.pa_mat.values))  # 平均出行时间(估计值)
            mean_tc = (np.sum((true_pa * self.cost).values)) / (np.sum(true_pa.values))  # 平均出行时间(真实值)
            loss = np.abs(mean_ec - mean_tc) / mean_tc  # 损失函数，平均阻抗的相对误差
            if (loss <= e) or (self.n >= max_iter):
                print('参数标定值：', params, '\n相对误差：', loss, '\n迭代次数：', self.n)
                return params
            else:
                for k in params.keys():
                    params[k] *= 1 + lr * loss

    def average_grow(self):
        ar, pr = np.meshgrid(self.zone_data[self.gj], self.zone_data[self.gi])
        g = (ar + pr) / 2  # 各位点的平均增长率
        return g

    def fratar(self):
        ar, pr = np.meshgrid(self.zone_data[self.gj], self.zone_data[self.gi])  # G
        a, p = np.meshgrid(self.pa_mat.sum(axis=0), self.pa_mat.sum(axis=1))  # T
        li = p[:, 0] / ((self.pa_mat * ar).sum(axis=1))  # Li
        lj = a[0, :] / ((self.pa_mat * pr).sum(axis=0))  # Lj
        lj, li = np.meshgrid(lj, li)
        g = ar * pr * (li + lj) / 2
        return g


class Spliter:
    def __init__(self, od: Union[pd.DataFrame, np.ndarray], v_list: [Union[pd.DataFrame, np.ndarray]]):
        self.od = od
        self.v_list = v_list  # 效用矩阵列表

    @staticmethod
    def v_mat(x_dict: dict, coef_dict: dict, const=0):  # 获取某一出行类型的效用矩阵
        one = np.ones_like(list(x_dict.values())[0])  # 常数项矩阵
        v = np.zeros_like(list(x_dict.values())[0])  # 初始化V矩阵
        for k in x_dict:
            v += x_dict[k] * coef_dict[k]  # 获取乘系数后的某项值
        v += const * one
        return v

    def logit(self, mu=1):
        vt = np.array(self.v_list)  # 获取效用的三维张量
        evs = np.exp(mu*vt)  # e^muV
        ev_sum = np.sum(evs, axis=0)  # sum e^muV
        n_list = []
        for i in self.v_list:  # 对每种出行类型
            p = np.exp(i) / ev_sum
            n_list.append(self.od * p)  # 求该出行方式下的OD矩阵
        return n_list


class Assigner:
    def __init__(self, g: nx.Graph, od: pd.DataFrame, cost='cost'):
        self.g = g
        self.od = od
        self.cost = cost

    def all_or_no(self, o, d, your_graph: Union[nx.Graph, None] = None, q_mane='x'):
        if your_graph is None:
            graph = self.g.copy()  # 创建副本
        else:
            graph = your_graph.copy()
        nx.set_edge_attributes(graph, 0, name=q_mane)
        q = self.od.loc[o, d]  # 获取要分配的流量
        path = nx.shortest_path(graph, source=o, target=d, weight=self.cost)
        edge_lst = net.poi2e(path, graph, mode='pp')
        for e in edge_lst:
            nx.set_edge_attributes(graph, {e: q}, name=q_mane)  # 分配流量至路径中每一个路段
        return graph

    def pma(self, o, d, theta=1):
        graph = self.g.copy()
        all_c = []  # 全部路径阻抗列表
        q = self.od.loc[o, d]  # 待分配的流量
        nx.set_edge_attributes(graph, 0, name='x')
        for i in list(nx.all_simple_edge_paths(graph, source=o, target=d)):
            path_c = []  # 路径总阻抗列表
            for e in i:
                path_c.append(graph.edges[e][self.cost])
            all_c.append(sum(path_c))

        all_c = np.exp(-theta * np.array(all_c))  # 似然值表
        sum_e = all_c.sum()  # 似然值求和

        n = 0  # 第n条路径
        for i in list(nx.all_simple_edge_paths(graph, source=o, target=d)):
            for e in i:
                orig = graph.edges[e]['x']
                nx.set_edge_attributes(graph, {e: orig + q * all_c[n] / sum_e}, name='x')
            n += 1
        return graph

    def dial(self, o, d, theta=1):
        graph = self.g.copy().to_directed()
        q = self.od.loc[o, d]
        # 初始化
        x = nx.shortest_path_length(graph, source=o, weight=self.cost)
        y = nx.shortest_path_length(graph, source=d, weight=self.cost)
        nx.set_node_attributes(graph, x, name='r')  # 添加节点r值
        nx.set_node_attributes(graph, y, name='s')  # 添加节点s值
        nx.set_edge_attributes(graph, 0, name='W')  # 初始化权重
        nx.set_edge_attributes(graph, 0, name='L')  # 初始化似然值
        nx.set_edge_attributes(graph, 0, name='x')  # 初始化流量

        # 计算似然值
        for e in graph.edges:
            ei, ej = e[0], e[1]  # 该边的端点
            ri, rj = graph.nodes[ei]['r'], graph.nodes[ej]['r']
            cij = graph.edges[e][self.cost]

            if ri < rj:
                nx.set_edge_attributes(graph, {(ei, ej): np.exp(theta * (rj - ri - cij))}, name='L')
            else:
                nx.set_edge_attributes(graph, {(ei, ej): 0}, name='L')

        # 向前计算
        for i in x.keys():  # 按r上升的顺序迭代
            if i == d:  # 到达终点时终止计算
                break
            if i == o:
                out_edges = graph.out_edges(i)
                for j in out_edges:
                    nx.set_edge_attributes(graph, {j: graph.edges[j]['L']}, name='W')
            else:
                # sum W
                w_sum = []
                in_edges = graph.in_edges(i)
                for in_e in in_edges:
                    w_sum.append(graph.edges[in_e]['W'])

                # W for every j
                out_edges = graph.out_edges(i)
                for j in out_edges:
                    w = graph.edges[j]['L'] * sum(w_sum)
                    nx.set_edge_attributes(graph, {j: w}, name='W')

        # 向后计算
        for j in y.keys():  # 按s上升的顺序迭代
            if j == o:  # 到达起点时终止计算
                break

            elif j == d:
                in_edges = graph.in_edges(j)

                # sum W
                w_sum = []
                for in_e in in_edges:
                    w_sum.append(graph.edges[in_e]['W'])

                # X for j = s
                for i in in_edges:
                    x = q * graph.edges[i]['W'] / sum(w_sum)
                    nx.set_edge_attributes(graph, {i: x}, name='x')

            else:
                in_edges = graph.in_edges(j)
                out_edges = graph.out_edges(j)

                # sum W
                w_sum = []
                for in_e in in_edges:
                    w_sum.append(graph.edges[in_e]['W'])

                # sum X
                x_sum = []
                for out_e in out_edges:
                    x_sum.append(graph.edges[out_e]['x'])

                for i in in_edges:
                    x = graph.edges[i]['W'] * sum(x_sum) / sum(w_sum)
                    nx.set_edge_attributes(graph, {i: x}, name='x')
        return graph

    def fw(self, o, d, param_dict, free_t, road_typ, bg, mode='UE', conv_val=10e-4, max_iter=50):
        graph = self.g.copy()
        test_val = 1
        flag = True
        n = 0
        while flag and (n < max_iter):
            n += 1
            nx.set_edge_attributes(graph, 0, name='y')  # 清理y
            if n == 1:
                for e in graph.edges:  # 初始化BPR计算量(同时也是分配量)（含背景交通流）
                    nx.set_edge_attributes(graph, {e: graph.edges[e][bg]}, name='x')

                # 第一步，求解阻抗并进行全由全无分配
                for e in graph.edges:
                    w = graph.edges[e]['x']  # 获取BPR计算量
                    arg_k = graph.edges[e][road_typ]  # 参数组序号
                    t0 = graph.edges[e][free_t]  # 自由流时间
                    t = self.bpr(t0=t0, w=w, **param_dict[arg_k])  # 求解BPR
                    nx.set_edge_attributes(graph, {e: t}, name='BPR')  # 给每条边附上BPR的值

                # 使用全有全无更新分配量x，此时bg被更新掉了，至此，x将不再附带bg
                nx.set_edge_attributes(graph, 0, name='x')  # 清理含bg的x
                graph = self.all_or_no(o, d, your_graph=graph)

            # 更新阻抗
            for e in graph.edges:
                w = graph.edges[e]['x'] + graph.edges[e][bg]  # 获取上一步的BPR计算量（含bg）
                arg_k = graph.edges[e][road_typ]  # 参数组序号
                t0 = graph.edges[e][free_t]  # 自由流时间
                t = self.bpr(t0=t0, w=w, **param_dict[arg_k])  # 求解BPR
                nx.set_edge_attributes(graph, {e: t}, name='BPR')  # 给每条边附上BPR的值

            graph = self.all_or_no(o, d, your_graph=graph, q_mane='y')  # 使用全由全无分配流量y

            # 基于二分法求解移动步长
            def obj_func(f_lda):
                obj_lst = []
                for f_e in graph.edges:
                    f_x_before = graph.edges[f_e]['x']
                    f_x_after = graph.edges[f_e]['y']
                    f_w = f_x_before + f_lda * (f_x_after - f_x_before)  # 求解积分上限
                    f_t0 = graph.edges[f_e][free_t]  # 自由流时间
                    f_arg_k = graph.edges[f_e][road_typ]  # 参数组序号
                    f_t = self.bpr(t0=f_t0, w=f_w, **param_dict[f_arg_k])  # 求解BPR

                    if mode == 'UE':
                        obj_lst.append(f_t * (f_x_after - f_x_before))  # 单项目标函数
                    elif mode == 'SO':
                        tx = self.bpr(t0=f_t0, w=f_x_before, **param_dict[f_arg_k])
                        ty = self.bpr(t0=f_t0, w=f_x_after, **param_dict[f_arg_k])
                        margin = w * (ty - tx) / (f_x_after - f_x_before)  # 边际时间
                        obj_lst.append((f_t + margin) * (f_x_after - f_x_before))  # 单项目标函数
                return sum(obj_lst)

            lda = opt.bisection(obj_func, 0, 1)

            # 确定新的迭代点
            x_before_lst = []
            x_new_lst = []
            for e in graph.edges:
                x_before = graph.edges[e]['x']
                x_after = graph.edges[e]['y']
                x_new = x_before + lda * (x_after - x_before)
                x_before_lst.append(x_before)
                x_new_lst.append(x_new)
                nx.set_edge_attributes(graph, {e: x_new}, name='x')

            # 收敛性检验
            x_before_lst = np.array(x_before_lst)[:3]
            x_new_lst = np.array(x_new_lst)[:3]
            test_val = np.sqrt(np.sum((x_new_lst - x_before_lst) ** 2)) / np.sum(x_before_lst)
            if test_val <= conv_val:
                flag = False
        print('迭代次数为：', n)
        print('收敛准则为：', test_val)
        return graph

    @ staticmethod
    def bpr(t0, c, w, a=0.15, b=4):  # BPR函数
        bpr = t0 * (1 + a * (w / c) ** b)
        return bpr


def pa2od(pa):
    after = (pa.T + pa) / 2
    return after


