"""
@File : _geo.py
@Author : 杨与桦
@Time : 2023/10/27 8:55
"""
import numpy as np


class CoordTrans:
    def __init__(self, coord=False, lon=False, lat=False):
        self.coord = coord
        if not isinstance(coord, bool):
            self.lng = coord[:, 0]
            self.lat = coord[:, 1]
        elif not(isinstance(lon, bool) and isinstance(lat, bool)):
            self.lng = lon
            self.lat = lat
        self.a = 6378245.0  # 长半轴
        self.ee = 0.00669342162296594323  # 偏心率平方

    def wgs842gcj02(self):
        dlat = self.transform_lat(self.lng - 105.0, self.lat - 35.0)
        dlng = self.transform_lng(self.lng - 105.0, self.lat - 35.0)
        radlat = self.lat / 180.0 * np.pi
        magic = np.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * np.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * np.cos(radlat) * np.pi)
        mglat = self.lat + dlat
        mglng = self.lng + dlng
        return mglng, mglat

    @staticmethod
    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * np.sqrt(np.abs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 * np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
                np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
                np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
        return ret

    @staticmethod
    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 * np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * np.pi) + 40.0 * np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 * np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
        return ret
