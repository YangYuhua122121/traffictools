"""
@File : appendix.py
@Author : 杨与桦
@Time : 2024/06/27 13:55
"""
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Union


class Park:
    def __init__(self, name, n=1, a=0, w=2.6, d=6):
        self.name = name
        if a == 0:  # 并行式布置
            self.L = n * d
            self.W = w
        else:  # 垂直式或斜式
            a = np.deg2rad(a)
            x = w / np.sin(a)
            y = d * np.cos(a) + w * np.sin(a) - x
            self.L = n * x + y
            self.W = d * np.sin(a) + w * np.cos(a)
        self.A = self.L * self.W
        self.N = n
        self.a, self.w, self.d = a, w, d

    def geomodel(self, pos: Union[list, tuple], base=0, color='#0D98BA', detail=True):
        if not isinstance(pos, (list, tuple)):
            raise TypeError('参数pos应当是一个列表或元组，表示参考点的（x, y）坐标')

        if detail:
            element_lst = []
            for i in self.__elements():
                tmp = pos_trans(np.array(i), pos, base)
                element_lst.append(shapely.Polygon(tmp))
            geometry = shapely.MultiPolygon(element_lst)
        else:
            occupy = np.array([[0, 0],
                               [self.L, 0],
                               [self.L, self.W],
                               [0, self.W]])
            occupy = pos_trans(occupy, pos, base)
            geometry = shapely.Polygon(occupy)
        gm = gpd.GeoDataFrame({'name': self.name,
                               'type1': ['Appendix'],
                               'type2': ['Park'],
                               'color': [color],
                               'geometry': [geometry]})
        return gm

    def __elements(self):
        lst = []

        if self.a == 0:  # 并列式
            s1 = np.array([[0, 0],
                           [self.d, 0],
                           [self.d, self.w],
                           [0, self.w]])
            for i in range(self.N):
                tmp = s1 + i * np.array([[self.d, 0]])
                lst.append(list(tmp))
        else:  # 垂直式或斜式
            s1 = np.array([[0, self.w * np.cos(self.a)],
                           [self.w * np.sin(self.a), 0],
                           [self.w * np.sin(self.a) + self.d * np.cos(self.a), self.d * np.sin(self.a)],
                           [self.d * np.cos(self.a), self.W]
                           ])
            for i in range(self.N):
                tmp = s1 + i * np.array([[self.w / np.sin(self.a), 0]])
                lst.append(list(tmp))
        return lst


def pos_trans(xy_set, pos, base):
    x_max = xy_set[:, 0].max()
    y_max = xy_set[:, 1].max()
    trans_map = {0: np.array([[0, 0]]),
                 1: np.array([-x_max, 0]),
                 2: np.array([-x_max, -y_max]),
                 3: np.array([0, -y_max])}
    origin = np.array([pos]) + trans_map[base]
    return xy_set + origin

