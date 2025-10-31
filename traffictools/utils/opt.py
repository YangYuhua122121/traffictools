"""
@File : opt.py
@Author : 杨与桦
@Time : 2023/12/01 22:01
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from traffictools.graph import od, net
from traffictools.utils import myplot
from typing import Union


def bisection(func, a, b, tol=10e-4, max_iter=50):
    if np.abs(func(a)) <= tol:
        return a
    if np.abs(func(b)) <= tol:
        return b
    if func(a) * func(b) > 0:
        return ValueError('区间内无根')
    a1, b1 = a, b
    bc = ValueError('不知道发生了啥(～￣(OO)￣)ブ')
    for i in range(max_iter):
        bc = (a1 + b1) / 2
        if np.abs(func(bc)) <= tol:
            return bc
        elif func(a1) * func(bc) < 0:
            b1 = bc
        else:
            a1 = bc
    return bc


class NetPlan:
    def __init__(self, table: pd.DataFrame, col_dict: Union[dict, None] = None,
                 time_limit: Union[str, int] = 'auto', undetermined=False):
        """
进行网络计划的编制
        :param table: 工序信息清单
        :param col_dict: 列名映射{键：实际列名}，键包括：[id, pre, time, [a, b, m]]，
        分别指[工序编号，紧前工序，工序时间，[乐观时间，保守时间, 最可能时间]]
        :param time_limit: 工期，默认取最后事项的最早完成时间
        :param undetermined: 启用非肯定型网络，此时需要添加a、b、m参数。可省略time参数
        """
        # 指定关键参数的列名，定义数据容器
        self.col_dict = {'id': 'id', 'pre': 'pre', 'time': 'time', 'a': 'a', 'b': 'b', 'm': 'm'}
        if col_dict:
            self.col_dict.update(col_dict)
        self.__table = table

        # 预处理：去除空格、更改为英文逗号分隔
        self.__table[[self.col_dict['id'], self.col_dict['pre']]] = (
            self.__table)[[self.col_dict['id'], self.col_dict['pre']]].astype(str)
        self.__table = self.__table.copy()
        self.__table[self.col_dict['pre']] = self.__table[self.col_dict['pre']].str.replace(' ', '')
        self.__table[self.col_dict['pre']] = self.__table[self.col_dict['pre']].str.replace('，', ',')
        self.__table[self.col_dict['pre']] = self.__table[self.col_dict['pre']].str.replace('nan', 'source')

        # 简化工序组（去除子集）
        self.simple_table = self.__pre_simplify()  # 经简化后的表格

        # 构造紧后工序，以搜索平行边。同时将平行边构造为实线-虚线边
        self.succ_table = self.__successor()
        tmp_parallel = self.__parallel_dashed()
        self.__table = pd.concat([self.succ_table.drop_duplicates(subset=[self.col_dict['pre'], 'succ']),
                                  tmp_parallel]).reset_index(drop=True)

        # 中间点个数(紧前工序组个数)
        self.mid_poi = len(self.__table[self.col_dict['pre']].drop_duplicates()) - 1  # 中间点个数（一组紧前工序确定一个事项）

        # 非肯定型网络参数计算
        if undetermined:
            self.__table['工序时间期望值'] =\
                ((self.__table[self.col_dict['a']] +
                  4 * self.__table[self.col_dict['m']] +
                  self.__table[self.col_dict['b']]) / 6)
            self.__table[self.col_dict['time']] = self.__table['工序时间期望值']
            self.__table['工序时间方差'] = ((self.__table[self.col_dict['b']] - self.__table[self.col_dict['a']]) / 6) ** 2
            self.col_dict['time'] = '工序时间期望值'
            self.tED = dict(
                zip(self.__table[self.col_dict['id']], self.__table[['工序时间期望值', '工序时间方差']].values.tolist()))

        # region 主流程
        # 排序
        pre_df = self.__sub_super_sort(self.__table)
        pre_df = self.__presort(pre_df)

        # 生成事项编号-紧前工序组的映射
        pre_list = pre_df[self.col_dict['pre']].drop_duplicates()
        self.__pre_no_dict = dict(zip(pre_list, [str(i) for i in range(1, self.mid_poi + 2)]))
        # 构造网络
        dash_edge = []  # 逻辑边(time=0)
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(['1', str(self.mid_poi + 2)])  # 初始化源、汇
        self.nx_graph.add_nodes_from([str(i) for i in range(2, self.mid_poi + 2)])  # 初始化中间节点
        # 实线连接，基于工序组完工事项（紧前工序-紧后工序）
        for i in pre_df.index:
            raw = pre_df.loc[[i], :]  # 工序i
            u = self.__pre_no_dict[list(raw[self.col_dict['pre']])[0]]  # 前端点为紧前工序
            v, sup = self.__pre_in_min_group(list(raw[self.col_dict['id']])[0],
                                             self.__pre_no_dict, str(self.mid_poi + 2))  # 后端点为最小紧后工序组
            dash_edge += sup
            self.nx_graph.add_edge(u, v, name=list(raw[self.col_dict['id']])[0], time=list(raw[self.col_dict['time']])[0],
                                   style='solid')  # 生成边，并指定工序名、权重与线型
        # 虚线连接，完善组（子工序组-父工序组）
        for dash in dash_edge:
            u, v = self.__pre_no_dict[dash[0]], self.__pre_no_dict[dash[1]]
            self.nx_graph.add_edge(u, v, time=0, style='dashed')
        # 事项参数计算
        self.__tek()
        self.tEk = nx.get_node_attributes(self.nx_graph, 'tE')  # 最早时间
        if time_limit == 'auto':
            self.TL = max(self.tEk.values())  # 总工期
        else:
            self.TL = time_limit
        self.__tlk()
        self.tLk = nx.get_node_attributes(self.nx_graph, 'tL')  # 最迟时间
        self.IJ = nx.get_edge_attributes(self.nx_graph, 'name')  # 各工序的起始点
        # 工序参数计算
        self.param_table = self.__tij()
        # endregion

        self.CP = list(self.param_table[self.param_table['关键工序'] == 1][self.col_dict['pre']])
        if undetermined:
            self.TLD = 0  # 工期方差
            for i in self.CP:
                self.TLD += self.tED[i][1]
            self.TLE = max(self.tEk.values())  # 工期期望
            self.__ud = undetermined

        # 修改列名
        self.param_table = (self.param_table[self.param_table['time'] != 0]
                            .rename(columns={'time': self.col_dict['time'], self.col_dict['pre']: self.col_dict['id']})
                            .reset_index(drop=True))

    # 紧前工序约简
    def __pre_simplify(self):
        new = []
        for i, j in self.__table.iterrows():
            pre_lst = j[self.col_dict['pre']].split(',')
            subset = []  # 子紧前工序
            if pre_lst[0] == 'source':
                new.append(j.to_frame().T)
            else:
                # 迭代工序组内每一个工序
                for m in pre_lst:
                    x = self.__table[self.__table[self.col_dict['id']] == m][self.col_dict['pre']].values[0]  # 子紧前工序
                    x = x.split(',')
                    subset += x
                j[self.col_dict['pre']] = ','.join(list(set(pre_lst) - set(subset)))
                new.append(j.to_frame().T)
        return pd.concat(new)  # 更新table

    # 紧后工序搜索
    def __successor(self):
        new = []
        for i, j in self.simple_table.iterrows():
            suss_lst_i = []
            for m, n in self.simple_table.iterrows():
                if j[self.col_dict['id']] in n[self.col_dict['pre']].split(','):  # 若j在某一工序的紧前工序中
                    suss_lst_i.append(n[self.col_dict['id']])
            j['succ'] = ','.join(suss_lst_i)
            new.append(j.to_frame().T)
        succ_dt = pd.concat(new)
        succ_dt['succ'] = succ_dt['succ'].replace('', 'sink')
        return succ_dt

    # 平行边搜索,紧前工序和紧后工序一致的为平行边
    def __parallel_dashed(self):
        # 平行边的清单（不包含主边）
        tmp1 = self.succ_table[self.succ_table.duplicated(subset=[self.col_dict['pre'], 'succ'])]
        tmp2 = []
        for i, j in tmp1.iterrows():
            j[self.col_dict['pre']] = j[self.col_dict['id']] + '_'
            tmp2.append(j.to_frame().T)
        for i, j in tmp1.iterrows():
            j[self.col_dict['id']] = j[self.col_dict['id']] + '_'
            j[self.col_dict['time']] = 0
            tmp2.append(j.to_frame().T)
        if len(tmp2) == 0:
            tmp1 = pd.DataFrame()
        else:
            tmp1 = pd.concat(tmp2).reset_index(drop=True)
        return tmp1

    # 基于工序组长排序，可以确保后驱节点的编号更大
    def __sub_super_sort(self, df: pd.DataFrame):
        # 计算每个工序组长
        len_list = []
        for i, j in df.iterrows():
            len_list.append(len(j[self.col_dict['pre']].split(',')))
        df['len'] = len_list
        return df.sort_values(by='len')

    # 判断改行数据是否具备完整的紧前工序
    def __has_pre(self, raw, lst):
        pres = raw[self.col_dict['pre']]
        pres = set(pres.split(','))  # 获取紧前工序组
        return pres.issubset(set(lst))  # 判断当前工序的紧前工序组是否为已有工序组的子集

    # 对工序列表进行排序，并对紧前工序组进行标号。##核心##
    def __presort(self, df: pd.DataFrame):
        df_init = df[df[self.col_dict['pre']] == 'source'].copy()  # 获取初始工序
        df_other = df[df[self.col_dict['pre']] != 'source']  # 获取其余工序

        for i in range(self.mid_poi):  # 迭代全部中间点
            id_tmp = list(df_init[self.col_dict['id']])  # 已有的前驱工序

            # 将准备完毕（紧前工序已加入init）的工序添加入init，并从other中删除
            for idx, j in df_other.iterrows():
                if self.__has_pre(j, id_tmp):
                    df_init = pd.concat([df_init, j.to_frame().T])
                    df_other = df_other.drop(index=idx)
        return df_init

    # 获取后驱节点点，对应最小紧后工序组/汇。##核心##
    def __pre_in_min_group(self, procedure: str, pre_no_map: dict, end):
        superset = []
        sup_res = []
        for i in pre_no_map.keys():
            if procedure in i.split(','):  # 将该工序对应的工序组添加至superset中
                superset.append([i, len(i.split(','))])  # 工序名组，组长
        superset.sort(key=lambda x: x[1])
        if len(superset) == 0:  # 若无最小紧后工序组，则取结点为后端点
            return end, []

        # 解析虚线
        for i in superset:
            sup_res.append(i[0])
        ds = []  # 虚线列表
        for i in sup_res[1:]:
            ds.append([sup_res[0], i])
        # 取最小工序组为后端点，并输出：子工序组——>父工序组列表
        return self.__pre_no_dict[superset[0][0]], ds

    # 事项参数计算-最早时间
    def __tek(self):
        self.nx_graph.nodes['1']['tE'] = 0  # 初始化起点最早时间

        # 按事项编号迭代，此时前驱工序一定完成
        for k in range(2, self.mid_poi + 3):
            node = str(k)  # 节点号（事项）
            pred_edge = dict(self.nx_graph.pred[node])  # 获取前驱节点及前驱边的属性
            te_lst = []  # 可选值
            for i in pred_edge.keys():  # i为前驱节点的编号
                pre_node_te = self.nx_graph.nodes[i]['tE']  # 前驱节点的te
                te_lst.append(pre_node_te + pred_edge[i]['time'])
            self.nx_graph.nodes[node]['tE'] = max(te_lst)  # 取最长路径

    # 事项参数计算-最迟时间
    def __tlk(self):
        self.nx_graph.nodes[net.spoi_search(self.nx_graph)[1][0]]['tL'] = self.TL  # 初始化终点的最迟时间

        # 按节点编号倒序进行。要求后驱节点全部具有tL值。可用反证法：
        # 若i的后驱编号k不具有tL值，则k<i。但标号规则又要求后驱编号k>i，矛盾
        for k in range(self.mid_poi + 1, 0, -1):
            node = str(k)  # 事项编号
            succ_edge = dict(self.nx_graph.succ[node])  # 后驱节点及后驱边的属性
            tl_lst = []  # 可选值
            for i in succ_edge.keys():  # i为后驱节点的编号
                succ_node_tl = self.nx_graph.nodes[i]['tL']  # 后驱节点的tL
                tl_lst.append(succ_node_tl - succ_edge[i]['time'])
            self.nx_graph.nodes[node]['tL'] = min(tl_lst)  # 取最长路径

    # 工序参数计算
    def __tij(self):
        tmp = net.net2df(self.nx_graph, mode='node')
        tmp1 = pd.merge(net.net2df(self.nx_graph), tmp[['name', 'tE']], left_on='O', right_on='name')
        tmp1 = pd.merge(tmp1, tmp, left_on='D', right_on='name')
        tmp1['最早完工时间'] = tmp1['tE_x'] + tmp1['time']
        tmp1['最迟开工时间'] = tmp1['tL'] - tmp1['time']
        tmp1['总时差'] = tmp1['最迟开工时间'] - tmp1['tE_x']
        tmp1['单时差'] = tmp1['tE_y'] - tmp1['最早完工时间']
        tmp1 = tmp1[['O', 'D', 'name_x', 'time', 'tE_x', 'tL', '最早完工时间', '最迟开工时间', '总时差', '单时差']]
        tmp1 = tmp1.rename(columns={'name_x': self.col_dict['pre'], 'tE_x': '最早开工时间', 'tL': '最迟完工时间'})
        tmp1 = tmp1.copy()
        tmp1.loc[tmp1['总时差'] == tmp1['总时差'].min(), '关键工序'] = 1
        tmp1['关键工序'] = tmp1['关键工序'].fillna(0)
        tmp1['关键工序'] = tmp1['关键工序'].astype(int)
        return tmp1

    # 绘制网络图
    def net_draw(self, pos: dict, grid=False,
                 s_color='#E63946', m_color='#A8DADC',
                 node_font_size=12, edge_font_size=12,
                 width=1, node_size=500, figsize=(10, 7), dpi=80
                 ):
        node_color = 2 * [s_color] + self.mid_poi * [m_color]  # 生成颜色列表
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        nx.draw(self.nx_graph, pos=pos, with_labels=True, node_color=node_color, node_size=node_size, width=width,
                font_size=node_font_size,
                style=list(nx.get_edge_attributes(self.nx_graph, 'style').values()), ax=ax)
        nx.draw_networkx_edge_labels(self.nx_graph, pos=pos, edge_labels=nx.get_edge_attributes(self.nx_graph, 'name'),
                                     font_size=edge_font_size, ax=ax)

        if grid:
            plt.axis('on')
            plt.grid(alpha=0.3)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    # 绘制横道图
    def gantt_draw(self, day_line=0, dt=1, Dt=3, dy=1.5,
                   lw=1.1, fill_alpha=1, font_size=12,
                   early_color='#E63946', late_color='#457B9D', fill_color='#A8DADC', dl_color='#1f77b4',
                   figsize=(15, 7), dpi=150):
        myplot.zh_allow()
        max_y = 0
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        for i, j in self.param_table.iterrows():
            ax.text(x=j['最早开工时间']-dt/4, y=dy * i, s=j[self.col_dict['id']], fontsize=font_size, ha='right')
            ax.hlines(y=dy * i + 0.1, xmin=j['最早开工时间'], xmax=j['最早完工时间'], lw=lw, color=early_color)
            if j['关键工序'] == 1:
                ax.hlines(y=dy * i - 0.1, xmin=j['最早开工时间'], xmax=j['最迟完工时间'], lw=lw, color=early_color)
            else:
                ax.hlines(y=dy * i - 0.1, xmin=j['最迟开工时间'], xmax=j['最迟完工时间'], lw=lw, ls='--', color=late_color)
            max_y = dy * i
        ax.hlines(y=0, xmin=0, xmax=0, lw=lw, color=early_color, label='最早开工进度')
        ax.hlines(y=0, xmin=0, xmax=0, lw=lw, ls='--', color=late_color, label='最迟开工进度')
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 网格设置
        t = np.arange(0, self.TL + 1, dt)
        T = np.arange(0, self.TL + 1, Dt)
        ax.set_xticks(t, minor=True)
        ax.set_xticks(T)

        # 现在时间设置
        if day_line:
            ax.axvline(x=day_line, label='今日线', color=dl_color)
            plt.fill_between(range(day_line+1), (day_line+1) * [-dy], (day_line+1) * [max_y + dy],
                             alpha=fill_alpha, color=fill_color)
            plt.text(day_line, max_y + dy, str(day_line), fontsize=font_size)

        plt.grid(which='minor', axis='x', alpha=0.4)
        plt.grid(which='major', axis='x')
        plt.legend(loc='upper left')

    # 完工概率计算
    def prob_cal(self, time_demand):
        if self.__ud:  # 非肯定型
            import scipy.stats as stats
            x = (time_demand - self.TLE) / (self.TLD ** (1 / 2))  # 服从标准正态分布
            cdf = stats.norm.cdf(x)
            return cdf
        else:
            if time_demand > self.TL:
                return 0
            else:
                return 1
