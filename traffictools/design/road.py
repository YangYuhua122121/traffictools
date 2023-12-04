"""
@File : road.py
@Author : 杨与桦
@Time : 2023/10/17 21:31
"""
import math
import pandas as pd
from typing import Union


class VLine:
    def __init__(self, i: int, length: Union[None, int, float] = None, dsv: Union[None, int, float] = None, name=None):
        self.i = i / 100
        self.length = length
        self.dsv = dsv
        self.name = name

    def slope_len_limit(self):
        if not isinstance(self.length, (int, float)):
            raise TypeError('应设置VLine的length参数！')
        max_limit = Table.vl_max_limit.loc[self.dsv, abs(self.i)]
        min_limit = Table.vl_min_limit.loc[0, self.dsv]
        rt = 1 if min_limit <= self.length <= max_limit else 0
        rg = (min_limit, max_limit)
        return rt, rg


class VLines:
    def __init__(self, vls: list, dsv=None):
        self.vls = vls  # 坡段组
        self.dsv = dsv  # 路段设计速度，指定时将各段的设计速度均改正
        if self.dsv:
            for vl in self.vls:
                vl.dsv = self.dsv

    def slope_len_limit(self):
        ps = []
        na_len = 0
        for i in range(len(self.vls)):
            vl = self.vls[i]
            vl: VLine
            if not vl.length:  # 待计算
                na_len = Table.vl_max_limit.loc[vl.dsv, abs(vl.i)]
                ps.append(0)
                break
            f, lim = VLine.slope_len_limit(vl)
            if not f:
                return f'第{i+1}个VLine坡长已超限！'
            ps.append(vl.length/lim[1])
        na_max = round((1 - sum(ps)) * na_len, 3)
        return ps, na_max


class VCurve:
    def __init__(self, vl1: VLine, vl2: VLine, r):
        self.i1 = vl1.i  # 前坡坡度
        self.i2 = vl2.i  # 后坡坡度
        self.R = r  # 顶点半径
        self.L = r * (self.i1 - self.i2)
        self.T = self.L / 2
        self.E = self.T ** 2 / (2 * r)
        # 确定符号与判断凹凸性
        self.sgn = [1, -1][self.i1 - self.i2 > 0]
        self.CONCAVITY = ['凹', '凸'][self.i1 - self.i2 > 0]

    def stake_cal(self, **kwargs):
        c = kwargs['c']  # 变坡点桩号
        x = kwargs['x']  # 目标点桩号
        ch = kwargs['ch']  # 变坡点高程
        s, e = c - self.T, c + self.T  # 起始桩号
        # 确定目标点处于前坡还是后坡
        sgn = [-1, 1][x > c]
        i = [self.i1, self.i2][x > c]
        th = ch + sgn * abs(x - c) * i  # 切线高程
        # 改正值
        if s < x < c:
            dh = (x - s) ** 2 / (2 * self.R)
        elif c < x < e:
            dh = (x - e) ** 2 / (2 * self.R)
        else:
            dh = 0
        h = th + self.sgn * dh  # 目标高程
        return round(h, 3)


class Table:
    vl_max_limit = pd.DataFrame({0.03: {120: 900, 100: 1000, 80: 1100, 60: 1200},
                                 0.04: {120: 700, 100: 800, 80: 900, 60: 1000, 40: 1100, 30: 1100, 20: 1200},
                                 0.05: {100: 600, 80: 700, 60: 800, 40: 900, 30: 900, 20: 1000},
                                 0.06: {80: 500, 60: 600, 40: 700, 30: 700, 20: 800},
                                 0.07: {40: 500, 30: 500, 20: 600},
                                 0.08: {40: 300, 30: 300, 20: 400},
                                 0.09: {30: 200, 20: 300},
                                 0.10: {20: 200}})
    vl_min_limit = pd.DataFrame({120: [300], 100: [250], 80: [200], 60: [150], 40: [120], 30: [100], 20: [60]})


if __name__ == '__main__':
    v1 = VLine(6, 400)
    v2 = VLine(5, 900)
    v3 = VLine(7)
    vs = VLines([v1, v2, v3], dsv=30)
    print(vs.slope_len_limit())
