"""
@File : opt.py
@Author : 杨与桦
@Time : 2023/12/01 22:01
"""
import numpy as np


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
