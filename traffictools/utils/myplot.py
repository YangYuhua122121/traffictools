"""
@File : myplot.py
@Author : 杨与桦
@Time : 2024/04/04 14:55
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union
from matplotlib.figure import Figure
import matplotlib

def zh_allow(font: Union[str, int] = 'SimHei'):
    """
    允许绘图中出现中文
    :param font: 字体选择（基于编号或名称）；字体库可见函数的返回值
    :return:
    """
    font_family1 = ['黑体', '微软雅黑', '微软正黑', '新宋体', '新细明体', '细明体', '仿宋', '楷体']
    font_family2 = ['SimHei', 'Microsoft YaHei', 'Microsoft JhengHei', 'NSimSun', 'PMingLiU',
                    'MingLiU', 'FangSong', 'KaiTi']
    font_family = pd.DataFrame({'编码': range(len(font_family1)), '字体名': font_family1, 'mpl代号': font_family2})
    if isinstance(font, int):
        font = font_family.loc[font, 'mpl代号']
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    plt.rcParams["font.sans-serif"] = [font]  # 设置字体的非sns方式
    return font_family

class InterFig:
    def __init__(self, fig: Figure, relim_k: float = 0.1, scale_k: float = 0.1):
        """
        将Figure对象转换为一个可交互的图窗。用户可继续在原Axes对象上绘图。

        ！！注意！！
        使用该功能时，应在导入matplotlib后马上设置后端：
        import matplotlib
        matplotlib.use('TkAgg')  # 或matplotlib.use('Qt5Agg')

        支持的交互方式包括：
        1. 鼠标左键拖动
        2. 鼠标滚轮放缩
        3. ↑↓←→键修改取值范围
        :param fig: 需要实现交互功能的Figure图窗
        :param relim_k: 修改取值范围的幅度，默认值为0.1
        :param scale_k: 放缩的幅度，默认值为0.1

        -----------------------------------
        Example:
        fig, ax = plt.subplots()  # 创建Figure图窗及Axes子图
        ax.scatter(range(0, 10000), range(0, 10000))  # 在子图上绘制
        ifig = InterFig(fig)  # 生成可交互图窗
        ifig.show()  # 渲染并作图（代替plt.show())

        """
        # try:
        #     matplotlib.use('TkAgg')
        # except ImportError:
        #     matplotlib.use('Qt5Agg')
        # except Exception as exp:
        #     print(exp)
        self._startx = 0
        self._starty = 0
        self._mPress = False
        self.fig = fig

        # resize幅度
        # self.ori_w, self.ori_h = self.fig.get_size_inches()
        # self._w_step = self.ori_w * relim_k
        # self._h_step = self.ori_h * relim_k

        # relim幅度
        self._relim_k = relim_k

        # scale幅度
        self._scale_k = 1 + scale_k

    def _call_move(self, event):
        if event.name == 'button_press_event':
            axtemp = event.inaxes
            if axtemp and event.button == 1:
                self._mPress = True
                self._startx = event.xdata
                self._starty = event.ydata
        elif event.name == 'button_release_event':
            axtemp = event.inaxes
            if axtemp and event.button == 1:
                self._mPress = False
        elif event.name == 'motion_notify_event':
            axtemp = event.inaxes
            if axtemp and event.button == 1 and self._mPress:
                x_min, x_max = axtemp.get_xlim()
                y_min, y_max = axtemp.get_ylim()
                w = x_max - x_min
                h = y_max - y_min
                mx = event.xdata - self._startx
                my = event.ydata - self._starty
                axtemp.set(xlim=(x_min - mx, x_min - mx + w))
                axtemp.set(ylim=(y_min - my, y_min - my + h))
                self.fig.canvas.draw_idle()  # 绘图动作实时反映在图像上
        return

    def _call_scroll(self, event):
        axtemp = event.inaxes
        # 计算放大缩小后， xlim 和ylim
        if axtemp:
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            w = x_max - x_min
            h = y_max - y_min
            curx = event.xdata
            cury = event.ydata
            curXposition = (curx - x_min) / w
            curYposition = (cury - y_min) / h
            if event.button == 'down':
                w = w * self._scale_k
                h = h * self._scale_k
            elif event.button == 'up':
                w = w / self._scale_k
                h = h / self._scale_k
            newx = curx - w * curXposition
            newy = cury - h * curYposition
            axtemp.set(xlim=(newx, newx + w))
            axtemp.set(ylim=(newy, newy + h))
            self.fig.canvas.draw_idle()  # 绘图动作实时反映在图像上
        return

    def _call_relim(self, event):
        """处理键盘事件 - 仅控制X/Y轴最大值"""
        # 获取当前范围
        ax = self.fig.axes[0]
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()

        # 计算当前轴长度
        x_length = x_right - x_left
        y_length = y_top - y_bottom

        # 右箭头：增大X最大值（保持X最小值不变）
        if event.key == 'right':
            new_x_right = x_right + x_length * self._relim_k
            ax.set_xlim(x_left, new_x_right)

        # 左箭头：减小X最大值（保持X最小值不变）
        elif event.key == 'left':
            new_x_right = x_right - x_length * self._relim_k
            ax.set_xlim(x_left, new_x_right)

        # 上箭头：增大Y最大值（保持Y最小值不变）
        elif event.key == 'up':
            new_y_top = y_top + y_length * self._relim_k
            ax.set_ylim(y_bottom, new_y_top)

        # 下箭头：减小Y最大值（保持Y最小值不变）
        elif event.key == 'down':
            new_y_top = y_top - y_length * self._relim_k
            ax.set_ylim(y_bottom, new_y_top)

        # 空格键重置视图
        elif event.key == ' ':
            ax.relim()
            ax.autoscale_view()

        # 更新范围显示并重新绘制
        self.fig.canvas.draw_idle()

    # def _call_resize(self, event):
    #     """处理键盘事件 - 调整图形尺寸"""
    #     # 获取当前图形尺寸
    #     current_width, current_height = self.fig.get_size_inches()
    #
    #     # 最小尺寸限制（避免图形过小）
    #     min_width = 2
    #     min_height = 2
    #
    #     # 根据按键调整图形尺寸
    #     if event.key == 'right':
    #         # 增加宽度
    #         new_width = current_width + self._w_step
    #         self.fig.set_size_inches(new_width, current_height)
    #     elif event.key == 'left':
    #         # 减小宽度（不小于最小值）
    #         new_width = max(min_width, current_width - self._w_step)
    #         self.fig.set_size_inches(new_width, current_height)
    #     elif event.key == 'down':
    #         # 增加高度
    #         new_height = current_height + self._h_step
    #         self.fig.set_size_inches(current_width, new_height)
    #     elif event.key == 'up':
    #         # 减小高度（不小于最小值）
    #         new_height = max(min_height, current_height - self._h_step)
    #         self.fig.set_size_inches(current_width, new_height)
    #     elif event.key == ' ':  # 空格键重置为初始尺寸
    #         self.fig.set_size_inches(self.ori_w, self.ori_h)
    #
    #     # 更新尺寸显示并重新绘制布局
    #     self.fig.tight_layout()  # 重新调整布局以适应新尺寸
    #     self.fig.canvas.draw_idle()

    def show(self) -> None:
        """
        InterFig的绘图渲染函数，用于代替plt.show()，以实现交互式绘图
        """
        self.fig.canvas.mpl_connect('scroll_event', self._call_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._call_move)
        self.fig.canvas.mpl_connect('button_release_event', self._call_move)
        self.fig.canvas.mpl_connect('motion_notify_event', self._call_move)
        # self.fig.canvas.mpl_connect('key_press_event', self._call_resize)
        self.fig.canvas.mpl_connect('key_press_event', self._call_relim)
        plt.show()
