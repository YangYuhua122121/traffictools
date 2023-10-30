"""
@File : bg.py
@Author : 杨与桦
@Time : 2023/10/05 20:00
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import Axes
class BitmapBG:
    def __init__(self, file:str):
        self.file = file
        self.img = plt.imread(file)
        self.img = self.img[::-1]    # 垂直镜像，转换为一般坐标轴
        self.v0 = None
    @staticmethod
    def LL2XY(lon, lat):
        # 采用墨卡托投影
        R = 6381372
        lon = np.radians(lon)
        lat = np.radians(lat)
        x = R * lon
        y = R*np.log(np.abs(np.tan(lat/2+np.pi/4)))
        return (x, y)

    @staticmethod
    def modulus(p1, p2):
        return  np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    # def theta(self, v1, v2):
    #     vz1 = np.array(list(v1)+[0])
    #     vz2 = np.array(list(v2)+[0])
    #     z = np.cross(vz1, vz2)  # 叉乘计算旋转方向
    #
    #     v1 = np.array(v1)
    #     v2 = np.array(v2)
    #     cos = np.dot(v1, v2)/(self.modulus(v1, (0, 0)) * self.modulus(v2, (0, 0)))
    #
    #     return np.arccos(cos)*z[2]/np.abs(z[2])  # 角度 * 旋转方向
    def point_locate(self,  pos1=None, pos2=None, plot=True,alpha=0.7, pc='red', lc='#1f77b4'):
        xr = self.img.shape[1]
        yr = self.img.shape[0]
         # 默认值
        if not pos1 :
            pos1 = (0.33, 0.33)
            pos1 = (xr*pos1[0], yr*pos1[1])
        if not pos2:
            pos2 = (0.66, 0.66)
            pos2 = (xr * pos2[0], yr * pos2[1])
        if plot:  # 作图定位
            plt.vlines(pos1[0], ymin=0, ymax=yr, colors=lc)
            plt.vlines(pos2[0], ymin=0, ymax=yr, colors=lc)
            plt.hlines(pos1[1], xmin=0, xmax=xr, colors=lc)
            plt.hlines(pos2[1], xmin=0, xmax=xr, colors=lc)
            plt.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c=pc)
            plt.imshow(self.img, origin='lower', alpha=alpha)
            ax = plt.gca()
            plt.minorticks_on()
            print('————————————微调建议————————————')
            print('x轴上移动:',int((ax.xaxis.get_minorticklocs()[1]-ax.xaxis.get_minorticklocs()[0])/2))
            print('y轴上移动:', int((ax.yaxis.get_minorticklocs()[1] - ax.yaxis.get_minorticklocs()[0])/2))
            print('——————建议值为副刻度线的一半——————')
            plt.grid(True)
            plt.show()
        self.v0 = [pos1, pos2]

    def background_locate(self, ax: Axes, pos1: tuple, pos2: tuple, alpha=0.5):
        # 基本参数
        height = self.img.shape[0]
        width = self.img.shape[1]
        pos1 = self.LL2XY(pos1[1], pos1[0])
        pos2 = self.LL2XY(pos2[1], pos2[0])
        print(pos1, pos2)

        # 旋转
        # angle = self.theta(np.array(self.v0[1]) - np.array(self.v0[0]), np.array(pos2) - np.array(pos1))
        # sinx, cosx = np.sin(angle), np.cos(angle)
        # new_height = int(np.abs(width*sinx)+np.abs*(height*cosx))
        # new_width = int(np.abs(width*cosx) + np.abs(height*sinx))  # 旋转后的新尺寸
        # R = np.array([[cosx, -sinx],
        #               [sinx, cosx]])

        # bg × scale = true
        scale = self.modulus(pos1, pos2) / self.modulus(self.v0[0], self.v0[1])
        # 新的原点：(x1true - scale × x1bg)
        new_ox = pos1[0] - scale * self.v0[0][0]
        new_oy = pos1[1] - scale * self.v0[0][1]
        # 新的宽高
        new_height = height*scale
        new_width = width*scale

        ax.imshow(self.img, origin='lower', alpha=alpha, zorder=-1, extent=[new_ox, new_ox + new_width, new_oy, new_oy + new_height])
        # ax.scatter([pos1[1], pos2[1]], [pos1[0], pos2[0]], c=['r', 'b'])
if __name__ == '__main__':
    f = 'C:/Users/杨/Desktop/map.png'
    fig, ax = plt.subplots()
    bg = BitmapBG(f)
    bg.point_locate((171, 236), (238, 559),plot=False)
    bg.background_locate(ax, (31.316785,121.219305), (31.373161,121.232765))
    plt.show()