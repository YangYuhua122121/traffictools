import os
import pandas as pd
import datetime
from collections.abc import Iterable
from typing import Union

class ResAnalyzer:
    TYPE_DETECTOR_DATA_COLLECTOR = '数据采集点'
    TYPE_DETECTOR_QUEUE_COUNTER = '排队计数器'
    TYPE_DETECTOR_TRAVEL_DETECTOR = '行程时间检测器'
    TYPE_DATA_SAMPLE = '样本数据'
    TYPE_DATA_AGG = '集计数据'

    def __init__(self, folder: str, detector_typ: str, data_typ: str, auto_latest_get: bool = False):
        """
        对TESS NG的数据结果展开分析。通过实例化一个分析器可以读取并合并TESS NG的输出结果
        任何针对输出结果的二次处理，均可以在self.data上进行，并重新赋值给self.data，以便数据同时支持客制化处理及ResAnalyzer的自带的处理方法
        **应注意，不宜修改ResAnalyzer相关方法所产生的列名，避免分析器无法解析数据**

        =========
        :param folder:目标数据结果的data文件夹（类似‘20250923202206’命名，含某次仿真所有检测器数据）或tess文件夹（类似'路网.tess'的命名，含某一路网对应的所有data文件夹）
        :param detector_typ:选择读取数据的检测器，输入值使用类属性ResAnalyzer.TYPE_DETECTOR_
        :param data_typ:选择读取数据的类型，输入值使用类属性ResAnalyzer.TYPE_DATA_
        :param auto_latest_get:默认为True，自动获取最新生成的数据。为True时，target_folder参数应为tess文件夹
        """
        # 基本信息
        self.detector_type = detector_typ
        self.data_type = data_typ
        self.folder = folder

        # 验证文件夹是否输入正确
        if auto_latest_get:  # 应输入tess文件夹
            if 'tess' not in os.path.basename(folder):
                raise ValueError(f'输入的文件夹{folder}不为tess文件夹！当auto_get=True时，应输入tess文件夹以便分析器读取最新的数据')
            else:
                folder = self._latest_folder_get()
        else:
            if 'tess' in os.path.basename(folder):
                raise ValueError(f'输入的文件夹{folder}不为data文件夹！当auto_get=False时，应输入data文件夹以便分析器链接到具体数据')

        # 构造合并数据
        lst = []

        for r, d, fs in os.walk(folder):
            for f in fs:
                file_info = f.split('_')
                if (file_info[0] == detector_typ) and (file_info[1] == data_typ):
                    file_path = os.path.join(r, f)
                    tmp = pd.read_csv(file_path, encoding='gbk')

                    if len(tmp) == 0:
                        print(f'文件{file_path}为空，已自动排除')
                    else:
                        tmp['id'] = f.split('ID')[1][:-5]
                        tmp['name'] = f.split('_')[2].split('(')[0]
                        lst.append(tmp)

        self.data: pd.DataFrame = pd.concat(lst).reset_index(drop=True)
        print(f'已完成data文件夹{folder}数据的读取与合并！')


    def data_get(self) -> pd.DataFrame:
        """
        获取读取合并的数据，该方法将返回一个self.data的副本。对返回值的操作无法直接影响self.data本身。若要修改self.data，应使用self.data
        :return:
        """
        return self.data.copy()

    def group_filter(self, ids: Iterable, group_name: str | int = None) -> pd.DataFrame:
        """
        指定检测器组的id列表，筛选出同组的检测器数据
        :param ids: 检测器组的id列表
        :param group_name: 可选，指定改组的组名
        :return:
        """
        detector_group = list(map(str, ids))
        tmp = self.data[self.data['id'].isin(detector_group)].copy()
        tmp['group_name'] = group_name
        return tmp

    def real_time_cal(self, start_time: pd.Timestamp, is_start0: bool = False) -> pd.DataFrame:
        """
        为self.data设置真实时间，同时返回一个self.data
        :param start_time: 真实起点时间，对应了仿真的起始时刻。请使用pd.to_datetime()转换为Timestamp类型
        :param is_start0: 默认为False，TESS NG结果数据的"起始时间(s)"为1s，启用该项时将其修改为0s
        :return:
        """
        time_columns = self._time_field()
        for c in time_columns:
            if (c == '起始时间(s)') and is_start0:
                self.data[c] -= 1
            self.data[f'真实_{c}'[:-3]] = pd.to_timedelta(self.data[c], unit='s') + start_time
        return self.data

    def _time_field(self) -> list[str]:
        if self.data_type == self.TYPE_DATA_AGG:
            return ['起始时间(s)', '结束时间(s)']
        else:
            if self.detector_type == self.TYPE_DETECTOR_DATA_COLLECTOR:
                return ['仿真时间(s)']
            elif self.detector_type == self.TYPE_DETECTOR_QUEUE_COUNTER:
                return ['起始时间(s)', '结束时间(s)', '采集时间(s)']
            else:
                return ['起始时间(s)', '结束时间(s)', '检测起始时间(s)', '检测结束时间(s)']

    def _latest_folder_get(self) -> str:
        # 获取父目录下的所有条目
        entries = os.listdir(self.folder)
        # 存储所有子文件夹及其创建时间
        folders = []
        for entry in entries:
            entry_path = os.path.join(self.folder, entry)
            # 检查是否为文件夹
            if os.path.isdir(entry_path):
                # 获取文件夹的创建时间
                # 在Windows上使用getctime()获取创建时间
                create_time = os.path.getctime(entry_path)
                # 转换为datetime对象
                create_datetime = datetime.datetime.fromtimestamp(create_time)
                folders.append((entry_path, create_datetime))

        if not folders:
            return ''

        # 按创建时间排序，最新的在最后
        folders.sort(key=lambda x: x[1])

        # 返回最新的文件夹路径
        return folders[-1][0]
