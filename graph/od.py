"""
@File : od.py
@Author : 杨与桦
@Time : 2024/05/20 20:01
"""


def poi2od(path_pois: list):
    """
将路径点转化为路径边（输入为列表）
    :param path_pois:
    :return:
    """
    path_ods = []
    for i in range(len(path_pois) - 1):
        path_ods.append(path_pois[i: i + 2])
    return path_ods
