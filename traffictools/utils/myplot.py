"""
@File : myplot.py
@Author : 杨与桦
@Time : 2024/04/04 14:55
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union


def zh_allow(font: Union[str, int] = 'SimHei'):
    font_family1 = ['黑体', '微软雅黑', '微软正黑', '新宋体', '新细明体', '细明体', '仿宋', '楷体']
    font_family2 = ['SimHei', 'Microsoft YaHei', 'Microsoft JhengHei', 'NSimSun', 'PMingLiU',
                    'MingLiU', 'FangSong', 'KaiTi']
    font_family = pd.DataFrame({'编码': range(len(font_family1)), '字体名': font_family1, 'mpl代号': font_family2})
    if isinstance(font, int):
        font = font_family.loc[font, 'mpl代号']
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    plt.rcParams["font.sans-serif"] = [font]  # 设置字体的非sns方式
    return font_family
