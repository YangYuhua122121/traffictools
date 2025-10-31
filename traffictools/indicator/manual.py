"""
@File : manual.py
@Author : 杨与桦
@Time : 2024/04/18 23:20
"""
import os
import pandas as pd
import numpy as np


class Sq:
    def __init__(self, name, direction: str = 'S'):
        self.ST = 1650
        self.SL = 1550
        self.SR = 1550
        self.STS = 3600
        self.STD = 1600
        self.I = 4
        self.fold = 'Sq'
        self.KSI = {1: 1, 2: 0.625, 3: 0.51, 4: 0.44}
        self.FPL = {60: 0.88, 90: 0.87, 120: 0.86}
        self.FPH = {60: 0.42, 90: 0.38, 120: 0.36}
        self.__direction = direction.upper()
        self.__filename = name

    def form_generate(self):
        general_col = ['id', 'W', 'G', 'HV', 'C', 'j']
        s_col = ['ge(可缺省)', 'bL(二选一)', 'qL(二选一)']
        l_col = ['n', 'qt0', 'lambda(可缺省)']
        r_col = ['r', 'p', 'bTD', 'bTS', 'Wb']
        t_col = ['qT', 'qturn']

        # 生成参数表
        if self.__direction == 'S':
            form = pd.DataFrame(columns=general_col + s_col)
        elif self.__direction == 'L':
            form = pd.DataFrame(columns=general_col + l_col)
        elif self.__direction == 'R':
            form = pd.DataFrame(columns=general_col + r_col)
        elif self.__direction == 'SL':
            form = pd.DataFrame(columns=general_col + s_col + l_col + t_col)
        elif self.__direction == 'SR':
            form = pd.DataFrame(columns=general_col + s_col + l_col + t_col)
        else:
            raise ValueError(f'direction参数设置不正确：{self.__direction} 不在[S、L、R、SL、SR]内')

        # 检索是否存在文件，若不存在则生成一个form
        try:
            pd.read_excel(f'./{self.fold}/{self.__filename}_{self.__direction}.xlsx')
        except OSError or FileNotFoundError:
            if not os.path.exists(f'./{self.fold}'):
                os.mkdir(f'{self.fold}')
            form.to_excel(f'./{self.fold}/{self.__filename}_{self.__direction}.xlsx', index=False)

    def conclude(self):
        # 列名
        output = ['饱和流量|pcu/h']
        f_general_col = ['id', 'fw', 'fg']
        f_s_col = ['fb']
        f_l_col = ['fL']
        f_r_col = ['fr', 'fp', 'fb_r', 'fpb']
        f_t_col = ['fTT']

        # 通用系数
        form = pd.read_excel(f'./{self.fold}/{self.__filename}_{self.__direction}.xlsx')
        form['fw'] = form['W'].apply(Sq.__fw)
        form['fg'] = form.apply(lambda x: Sq.__fg(x['G'], x['HV']), axis=1)

        # 直行车道求解过程
        def s_process(op=output[0]):
            form['fb'] = form.apply(lambda x: Sq.__fb(x['bL(二选一)'], x['ge(可缺省)'],
                                                      x['C'], x['j'], x['qL(二选一)'], self.I), axis=1)
            form[op] = (form['fw'] * form['fg'] * form['fb'] * self.ST).astype(int)

        # 左转车道求解过程
        def l_process(op=output[0]):
            form['fL'] = form.apply(lambda x: Sq.__fl(x['n'], x['qt0'], x['lambda(可缺省)'], self.KSI,
                                                      x['C'], x['j'], self.I), axis=1)
            form[op] = (form['fw'] * form['fg'] * form['fL'] * self.SL).astype(int)

        # 右转车道求解过程
        def r_process(op=output[0]):
            form['fr'] = form.apply(lambda x: Sq.__fr(x['r']), axis=1)
            form['fp'] = form.apply(lambda x: Sq.__fp(x['p'], x['C'], self.FPL, self.FPH), axis=1)
            form['fb_r'] = form.apply(lambda x: Sq.__fb_r(x['C'], x['bTD'], x['bTS'], self.STD, self.STS, x['Wb'])
                                      , axis=1)
            form['fpb'] = form[['fp', 'fb_r']].min(axis=1)
            form[op] = (form['fw'] * form['fg'] * form['fr'] * form['fpb'] * self.SR).astype(int)

        # 专用系数
        if self.__direction == 'S':
            s_process()
            res = form[f_general_col + f_s_col + output].copy()
        elif self.__direction == 'L':
            l_process()
            res = form[f_general_col + f_l_col + output].copy()
        elif self.__direction == 'R':
            r_process()
            res = form[f_general_col + f_r_col + output]
        elif self.__direction == 'SL':
            s_process('ST')
            l_process('SL')
            form['fTT'] = form.apply(lambda x: Sq.__ft_turn(x['qT'], x['qturn'], x['ST'], x['SL']), axis=1)
            form[output[0]] = (form['ST'] * form['fTT']).astype(int)
            res = form[f_general_col + f_s_col + f_l_col + f_t_col + output]
        elif self.__direction == 'SR':
            s_process('ST')
            r_process('SR')
            form['fTT'] = form.apply(lambda x: Sq.__ft_turn(x['qT'], x['qturn'], x['ST'], x['SR']), axis=1)
            form[output[0]] = (form['ST'] * form['fTT']).astype(int)
            res = form[f_general_col + f_s_col + f_r_col + f_t_col + output]
        else:
            res = -1
        return res

    # 车道宽度修正
    @staticmethod
    def __fw(w):
        if 3 <= w <= 3.5:
            return 1
        elif 2.7 <= w < 3:
            return 0.4 * (w - 0.5)
        elif w > 3.5:
            return 0.05 * (w + 16.5)
        else:
            raise ValueError(f'输入的车道宽度w不符合规范：(w = {w} < 2.7m)')

    # 坡度与大车修正
    @staticmethod
    def __fg(g, hv):
        if g < 0:  # 下坡时，取0
            g = 0
        if hv <= 0.5:
            return 1 - (g + hv)
        else:
            raise ValueError(f'大车率hv超出规范要求：hv = {hv} > 0.5')

    # 直行-非机动车修正
    @staticmethod
    def __fb(bl, ge, c, j, ql, i):
        if np.isnan(ge):
            ge = c / j - i
        if np.isnan(bl):
            bl = int(ql * (c - ge) / 3600)
        if bl > 0:
            return round(1 - (1 + np.sqrt(bl)) / ge, 3)
        else:
            return 1

    # 左转校正系数
    @staticmethod
    def __fl(n, qt0, _lambda, ksi, c, j, i):
        if np.isnan(_lambda):
            ge = c / j - i
            _lambda = ge / c
        if qt0 == 0:
            return 1
        else:
            n = ksi[n]
            return np.exp(-0.001 * n * qt0 / _lambda)

    # 右转-转弯半径校正系数
    @staticmethod
    def __fr(r):
        if r > 15:
            return 1
        elif 0 <= r <= 15:
            return 0.5 + r / 30
        else:
            ValueError(f'右转转弯半径r为负数：r = {r} < 0 ')

    # 右转-行人校正系数(近似)
    @staticmethod
    def __fp(p, c, fpl, fph):
        basic_c = np.array([60, 90, 120])
        c = basic_c[np.argmin(np.abs(c - basic_c))]  # 获取近似信号周期
        if p <= 20:
            return fpl[c]
        else:
            return fph[c]

    # 右转-非机动车校正系数
    @staticmethod
    def __fb_r(c, btd, bts, std, sts, wb):
        tt = 3600 * (bts / sts + btd / std) / wb
        return 1 - tt / c

    # 直转合流校正系数
    @staticmethod
    def __ft_turn(qt, q_turn, st, sturn):
        k = st / sturn
        qt1 = k * q_turn + qt
        return (qt + q_turn) / qt1


if __name__ == '__main__':
    pass
