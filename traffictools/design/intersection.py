"""
@File : intersection.py
@Author : 杨与桦
@Time : 2024/06/29 21:27
"""
import os
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import plotting
import matplotlib.pyplot as plt


class Cross:
    def __init__(self, name, fold='Intersection', stopline_b=1.5):
        self.name = name  # 文件名
        self.fold = fold  # 文件夹名

        self.stopline_b = stopline_b  # 停止线与人行道间距

        # 构造初始参数表格
        entry_uses = ['L, S, R'] * 4
        entry_ws = ['3.5, 3.5, 3.5'] * 4
        exit_ws = ['3.5, 3.5, 3.5'] * 4
        turn_rs = [10] * 4
        lens = [40] * 4
        alphas = [90] * 4
        slopes = [0] * 4
        pedcross_back = [4.5] * 4
        pedcross_w = [4] * 4
        form = pd.DataFrame([entry_uses, entry_ws, exit_ws, turn_rs, lens, alphas, slopes,
                             pedcross_back, pedcross_w],
                            columns=[0, 1, 2, 3],
                            index=['entry_use', 'entry_w', 'exit_w', 'turn_r', 'len', 'alpha', 'i',
                                   'pedcross_back', 'pedcross_w'])

        # 检索是否存在文件，若不存在则生成一个form
        try:
            form = pd.read_excel(f'./{self.fold}/Cross_{self.name}.xlsx', index_col=0)
        except OSError or FileNotFoundError:
            if not os.path.exists(f'./{self.fold}'):
                os.mkdir(f'{self.fold}')
            form.to_excel(f'./{self.fold}/Cross_{self.name}.xlsx', index=True)

        # 参数表格
        self.param_table = form
        self.__params = self.__param(form)

    def geomodel(self):
        a, b, c = self.__circle(self.__params)  # 转弯半径图形，转弯半径圆心坐标，交叉口菱形角点坐标
        corner_without_round = self.__corner(self.__params, b, c)
        corner = corner_without_round.union(shapely.MultiPolygon(a))  # 带转弯半径的街角

        rec = corner.bounds
        rec = shapely.Polygon([(rec[0], rec[1]), (rec[2], rec[1]),
                               (rec[2], rec[3]), (rec[0], rec[3])])  # 最大范围
        cross = rec.difference(corner)  # 交叉口路面

        ped_in, ped_out, ped_out_w = self.__ped_cross(self.__params, c)
        pedc = ped_out.difference(ped_in).intersection(cross)  # 人行道

        cline = self.__center_line(self.__params, rec, ped_out)

        stop_line = self.__stop_line(ped_out_w, cline, cross, corner, c)

        geo_sidewalk = gpd.GeoDataFrame({'name': [self.name],
                                         'type1': ['Intersection'],
                                         'type2': ['SideWalk'],
                                         'color': ['#A8DADC'],
                                         'edgecolor': ['#F1FAEE'],
                                         'linewidth': [1.5],
                                         'geometry': [corner.buffer(-0.5)]})
        geo_cross = gpd.GeoDataFrame({'name': [self.name],
                                      'type1': ['Intersection'],
                                      'type2': ['Cross'],
                                      'color': ['#708090'],
                                      'edgecolor': ['#F1FAEE'],
                                      'linewidth': [1.5],
                                      'geometry': [cross]})
        geo_pedcross = gpd.GeoDataFrame({'name': [self.name],
                                         'type1': ['Intersection'],
                                         'type2': ['PedCross'],
                                         'color': ['#F5F5F5'],
                                         'edgecolor': ['#1C3144'],
                                         'linewidth': [1.5],
                                         'geometry': [pedc]})
        geo_cline = gpd.GeoDataFrame({'name': [self.name],
                                      'type1': ['Intersection'],
                                      'type2': ['CenterLine'],
                                      'color': ['#F39C12'],
                                      'edgecolor': ['#FFFFFF00'],
                                      'linewidth': [1.5],
                                      'geometry': [cline]})
        geo_sline = gpd.GeoDataFrame({'name': [self.name],
                                      'type1': ['Intersection'],
                                      'type2': ['StopLine'],
                                      'color': ['#F1FAEE'],
                                      'edgecolor': ['#F1FAEE'],
                                      'linewidth': [2.5],
                                      'geometry': [stop_line]})
        gm = pd.concat([geo_cross, geo_sidewalk, geo_pedcross, geo_cline, geo_sline])
        return gm

    # 构造参数字典
    def __param(self, data):
        param1 = {'LH': self.__str2lst(data[0]['exit_w']), 'LV': self.__str2lst(data[1]['entry_w']),
                  'R': data[0]['turn_r'], 'A': np.deg2rad(data[0]['alpha']), 'sgn': 1,
                  'LEN': data[0]['len'], 'PB': data[0]['pedcross_back'], 'PW': data[0]['pedcross_w']}
        param2 = {'LH': self.__str2lst(data[2]['entry_w']), 'LV': self.__str2lst(data[1]['exit_w']),
                  'R': data[1]['turn_r'], 'A': np.deg2rad(180 - data[2]['alpha']), 'sgn': -1,
                  'LEN': data[1]['len'],'PB': data[1]['pedcross_back'], 'PW': data[1]['pedcross_w']}
        param3 = {'LH': -self.__str2lst(data[2]['exit_w']), 'LV': -self.__str2lst(data[3]['entry_w']),
                  'R': -data[2]['turn_r'], 'A': np.deg2rad(data[2]['alpha']), 'sgn': 1,
                  'LEN': data[2]['len'], 'PB': data[2]['pedcross_back'], 'PW': data[2]['pedcross_w']}
        param4 = {'LH': -self.__str2lst(data[0]['entry_w']), 'LV': -self.__str2lst(data[3]['exit_w']),
                  'R': -data[3]['turn_r'], 'A': np.deg2rad(180 - data[0]['alpha']), 'sgn': -1,
                  'LEN': data[3]['len'], 'PB': data[3]['pedcross_back'], 'PW': data[3]['pedcross_w']}
        return {'p1': param1, 'p2': param2, 'p3': param3, 'p4': param4}

    @staticmethod
    # 构造转弯半径，返回图形、圆心坐标、菱形坐标
    def __circle(param_dict: dict):
        shape_res = []
        xy_res = []
        rhom_res = []
        for i in param_dict.values():
            rhom_x = i['LH'] / np.sin(i['A']) + i['LV'] / np.tan(i['A'])
            rhom_y = i['sgn'] * i['LV']
            x = i['R'] / np.tan(i['A'] / 2) + rhom_x
            y = i['sgn'] * i['R'] + rhom_y
            shape_res.append(shapely.Point(x, y).buffer(np.abs(i['R'])))
            xy_res.append((x, y))
            rhom_res.append((rhom_x, rhom_y))
        return shape_res, xy_res, rhom_res

    @staticmethod
    # 构造四角面域
    def __corner(param_dict: dict, cc, rhom):
        # 道路长（辅助）
        lens = [param_dict['p1']['LEN'], param_dict['p3']['LEN'], -param_dict['p3']['LEN'], -param_dict['p1']['LEN']]
        alphas = [param_dict['p1']['A'], param_dict['p3']['A'], param_dict['p3']['A'], param_dict['p1']['A']]

        # 第一次塑形（三角）
        shape_res = []
        for i, j, k, n in zip(param_dict.values(), cc, rhom, range(4)):  # i为参数，j为圆心坐标，k为交叉口菱形四角坐标
            x1, y1 = j[0], j[1] - i['sgn'] * i['R']
            x2, y2 = k[0] + lens[n], k[1]
            x3, y3 = k[0] + lens[n] * np.cos(i['A']), k[1] + i['sgn'] * lens[n] * np.sin(i['A'])
            x4, y4 = j[0] - i['R'] * np.sin(i['A']), j[1] + i['R'] * np.cos(alphas[n])
            shape_res.append(shapely.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))

        # 边界坐标（辅助）
        bs = shapely.MultiPolygon(shape_res).bounds
        bs = [(bs[2], bs[3]),
              (bs[2], bs[1]),
              (bs[0], bs[1]),
              (bs[0], bs[3])]

        # 第二次塑形（四角）
        shape_res = []
        for i, j, k, b, n in zip(param_dict.values(), cc, rhom, bs, range(4)):  # i为参数，j为圆心坐标，k为交叉口菱形四角坐标，b为极值坐标
            x1, y1 = j[0], j[1] - i['sgn'] * i['R']
            x2, y2 = b[0], k[1]
            x3, y3 = b[0], b[1]
            x4, y4 = k[0] + lens[n] * np.cos(i['A']), k[1] + i['sgn'] * lens[n] * np.sin(i['A'])
            x5, y5 = j[0] - i['R'] * np.sin(i['A']), j[1] + i['R'] * np.cos(alphas[n])
            shape_res.append(shapely.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]))
        corner_shp = shapely.MultiPolygon(shape_res)
        return corner_shp

    # 构造人行横道，返回内外shp,并返回pbw(array)，用于停止线生成
    def __ped_cross(self, param_dict, prhom):
        pb = [param_dict['p1']['PB'], param_dict['p2']['PB'], param_dict['p3']['PB'], param_dict['p4']['PB']]
        pw = [param_dict['p1']['PW'], param_dict['p2']['PW'], param_dict['p3']['PW'], param_dict['p4']['PW']]
        pbw = np.array(pb) + np.array(pw)

        ped_inner = self.__rec_offset(prhom, list(pb))
        ped_outer = self.__rec_offset(prhom, list(pbw))  # 基于交叉口内菱形外扩
        return ped_inner, ped_outer, pbw

    # 构造中心线，该函数直接返回shapely对象
    def __center_line(self, param_dict, prec, pped_outer):
        inf_len = 1000
        hline = shapely.LineString([(-inf_len, 0), (inf_len, 0)])  # 水平中心线
        vline1 = shapely.LineString(
            [(0, 0), (inf_len * np.cos(param_dict['p1']['A']), inf_len * np.sin(param_dict['p1']['A']))])
        vline2 = shapely.LineString(
            [(0, 0), (-inf_len * np.cos(param_dict['p3']['A']), -inf_len * np.sin(param_dict['p3']['A']))])
        cline = shapely.MultiLineString([hline, vline1, vline2])  # 超长中心线
        cline = cline.intersection(prec)  # 限定于最大范围
        cline = cline.difference(pped_outer.buffer(self.stopline_b))  # 减去人行横道区域，并进一步后退
        return cline

    # 停止线
    def __stop_line(self, psw: np.ndarray, p_cline_shp, p_cross_shp, p_sidewalk_shp, p_rhom_xy):
        stop_rec = self.__rec_offset(p_rhom_xy, list(psw + self.stopline_b)).boundary  # 停车线(矩形)
        stop_rec = stop_rec.intersection(p_cross_shp)  # 获取每条接入道路的停车线（出口道+进口道）
        plotting.plot_line(stop_rec)

        # 延长中心线以确保相交
        cline_extend = []
        for i in p_cline_shp.geoms:
            cline_extend.append(self.__line_extend(i, 5))
        p_cline_shp = shapely.MultiLineString(cline_extend)
        plotting.plot_line(p_cline_shp)

        res_lst = []
        for i in stop_rec.geoms:
            i = self.__line_extend(i, 1)  # 停车线延长一个单位，以确保相交
            poi1 = i.intersection(p_cline_shp)
            poi2 = i.intersection(p_sidewalk_shp.boundary)
            print(poi1, poi2)
            res_lst.append(shapely.LineString([poi1, poi2]))

        # for i, j, k in [(0, 1, 0), (1, 3, 1), (2, 0, 2), (3, 2, 3)]:  # i停车线编号,j中心线编号,k路侧人行道编号
        #     i_stop_rec = self.__line_extend(stop_rec.geoms[i], 1)  # 停车线延长1个单位，以确保相交
        #     t1 = self.__line_extend(p_cline_shp.geoms[j], 1)  # 中心线延长1个单位，以确保相交
        #     t2 = p_sidewalk_shp.geoms[k].boundary  # 获取边
        #     poi1 = i_stop_rec.intersection(t1)
        #     poi2 = i_stop_rec.intersection(t2)
        #     res_lst.append(shapely.LineString([poi1, poi2]))
        return shapely.MultiLineString(res_lst)

    @staticmethod
    # 菱形区域外扩展，返回shapely
    def __rec_offset(rhom, offset):  # 菱形坐标，偏移距离
        xoffset = [offset[1], offset[1], -offset[3], -offset[3]]  # [+w2, +w2, -w4, -w4]
        yoffset = [offset[0], -offset[2], -offset[2], offset[0]]  # y=[+w1, -w3, -w3, +w1]
        new_rhom = []
        for p, i in zip(rhom, range(4)):
            new_rhom.append((p[0] + xoffset[i], p[1] + yoffset[i]))
        return shapely.Polygon(new_rhom)

    @staticmethod
    # 直线延长
    def __line_extend(line: shapely.LineString, a):
        x1, y1 = line.boundary.geoms[0].coords[0]
        x2, y2 = line.boundary.geoms[1].coords[0]

        if abs(x2 - x1) < 0.001:
            # 斜率无穷的情况
            dx = 0
            dy = a
        else:
            # 基于直线斜率计算dx、dy
            k = (y2 - y1) / (x2 - x1)
            dx = a / np.sqrt(k ** 2 + 1)
            dy = k * dx

        # 基于向量方向判断x、y的增量方向
        sgnx = np.sign(x1 - x2)
        sgny = np.sign(y1 - y2)
        x1, y1 = x1 + sgnx * dx, y1 + sgny * dy
        x2, y2 = x2 - sgnx * dx, y2 - sgny * dy

        return shapely.LineString([(x1, y1), (x2, y2)])

    @staticmethod
    # 将字符串转换为列表和
    def __str2lst(obj: str):
        obj = obj.replace(' ', '')
        obj = obj.replace('，', ',')
        return sum(list(map(float, obj.split(','))))
