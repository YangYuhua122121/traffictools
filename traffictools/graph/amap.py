"""
@File : amap.py
@Author : 杨与桦
@Time : 2023/10/26 15:54
"""
import requests
import pandas as pd
from typing import Union
from shapely import Polygon
from shapely import LineString
import geopandas as gpd


class AMap:
    def __init__(self, key: str):
        """
生成一个AMap对象，用于后续调用高德API
        :param key: 向高德申请的密钥
        """
        self.key = key

    def poi_search(self, types: list, mode: str = 'area',
                   area: Union[None, int, str] = None,
                   loc: Union[None, tuple] = None,
                   r: Union[None, int] = None,
                   poly: Union[None, list[tuple]] = None):
        """
兴趣点（POI）搜索
        :param types:类别列表
        :param mode:搜索模式，默认为‘area’
        :param area:搜索区划的id编码（area模式下）
        :param loc:搜索区域的圆心（around模式下）
        :param r:搜索区域的半径（around模式下）
        :param poly:搜索区域的边坐标（poly模式下）
        :return:
        """
        types = map(str, types)
        types = '|'.join(types)
        parameters = {'key': self.key,
                      'types': types,
                      'citylimit': 'true',
                      'offset': 25}
        info = ['name', 'type', 'typecode', 'address', 'location']

        if (mode == 'area') and area:
            parameters['city'] = area
            url = 'https://restapi.amap.com/v3/place/text'
        elif (mode == 'around') and loc and r:
            parameters['location'] = str(loc[0]) + ',' + str(loc[1])
            parameters['radius'] = r
            info.append('distance')
            url = 'https://restapi.amap.com/v3/place/around'
        elif (mode == 'poly') and poly:
            ps = ''
            for point in poly:
                tmp = str(point[0]) + ',' + str(point[1]) + '|'
                ps += tmp
            ps = ps[:-1]
            parameters['polygon'] = ps
            url = 'https://restapi.amap.com/v3/place/polygon'
        else:
            raise ValueError('参数输入错误！')

        page = 1
        data_lst = []
        while True:
            parameters['page'] = page
            res = requests.get(url=url, params=parameters)
            poi_data = res.json()['pois']
            if len(poi_data) == 0:
                break
            poi_data = pd.DataFrame(poi_data)[info]
            poi_data['type1'] = poi_data['type'].apply(lambda x: x.split(';')[0])
            poi_data['type2'] = poi_data['type'].apply(lambda x: x.split(';')[1])
            poi_data['type3'] = poi_data['type'].apply(lambda x: x.split(';')[2])
            poi_data['lon'] = poi_data['location'].apply(lambda x: x.split(',')[0])
            poi_data['lat'] = poi_data['location'].apply(lambda x: x.split(',')[1])
            poi_data['geometry'] = gpd.points_from_xy(poi_data['lon'], poi_data['lat'])
            poi_data.drop(columns=['type', 'location', 'lon', 'lat'], inplace=True)
            data_lst.append(poi_data)
            page += 1
        poi_data = pd.concat(data_lst).reset_index(drop=True)
        poi_data = gpd.GeoDataFrame(poi_data, crs='EPSG:4326')
        return poi_data

    def region_get(self, name: Union[int, str]):
        """
获取区划信息
        :param name:目标区划的id编码
        :return:
        """
        parameter = {'key': self.key,
                     'keywords': name,
                     'subdistrict': 0,
                     'extensions': 'all'}
        res = requests.get('https://restapi.amap.com/v3/config/district', params=parameter)
        data = res.json()['districts']
        data = pd.DataFrame(data)[['adcode', 'name', 'polyline']]
        s = data['polyline'][0]
        s: str
        ds = s.split('|')  # 多个地块
        d = ds[0].split(';')  # 单个地块
        poly_geometry = Polygon([(float(i.split(',')[0]), float(i.split(',')[1])) for i in d])
        data['polyline'] = poly_geometry
        data.rename(columns={'polyline': 'geometry'}, inplace=True)
        data = gpd.GeoDataFrame(data, crs='EPSG:4326')
        return data

    def route_plan(self, o: tuple, d: tuple, mode='car'):
        """
获取路径规划
        :param o:起点经纬度坐标
        :param d: 目的地经纬度坐标
        :param mode: 交通方式
        :return:
        """
        param_lst = ['walk', 'bus', 'car']
        url_list = ['walking', 'transit/integrated', 'driving']
        url_kw = url_list[param_lst.index(mode)]
        parameter = {'key': self.key,
                     'origin': str(o[0]) + ',' + str(o[1]),
                     'destination': str(d[0]) + ',' + str(d[1]),
                     }

        if mode == 'bus':
            # 确定公交起始的城市
            o_city, d_city = self.loc2name(o)['citycode'], self.loc2name(d)['citycode']
            parameter['city'] = o_city
            parameter['cityd'] = d_city

        res = requests.get(f'https://restapi.amap.com/v3/direction/{url_kw}', params=parameter)
        data = res.json()['route']

        if mode == 'bus':
            res_lst = []
            data = data['transits'][0]['segments']  # 当前直接选择第一种路线方案
            for i in range(len(data)):  # 获取每个segment
                segment = data[i]  # 某一个segment，即路段（无换乘）
                for j in segment:  # 获取每个segment下的特定移动方式
                    move_mode = segment[j]
                    if len(move_mode) != 0:  # 存在空的方式(如taxi经常空)
                        if j == 'walking':
                            tmp = pd.DataFrame(move_mode['steps'])[['polyline', 'distance']]
                            v = float(move_mode['distance'])/float(move_mode['duration'])  # 通过路程总参数计算速度
                            tmp['duration'] = tmp['distance'].astype(float)/v  # 计算小路段的用时
                            tmp = tmp.round(0)  # 将时间取整
                            tmp['type'] = '步行'
                            tmp['group'] = i  # 编组
                            res_lst.append(tmp)
                        elif j == 'bus':
                            for k in move_mode['buslines']:  # 含多个公交线路
                                tmp = pd.DataFrame({'polyline': [k['polyline']],
                                                    'type': [k['type']],
                                                    'distance': [k['distance']],
                                                    'duration': [k['duration']],
                                                    'group': [i]})
                                res_lst.append(tmp)
            df = pd.concat(res_lst)

        else:
            df = pd.DataFrame(data['paths'][0]['steps'])
            df = df[['polyline', 'distance', 'duration']]

        df['polyline'] = df['polyline'].apply(
            lambda r: [(float(i.split(',')[0]), (float(i.split(',')[1]))) for i in r.split(';')]
        )
        df['geometry'] = df['polyline'].apply(lambda r: LineString(r))
        df.drop(columns='polyline', inplace=True)
        df = gpd.GeoDataFrame(df, crs='EPSG:4326')

        return df

    def loc2name(self, loc: tuple):
        parameter = {'key': self.key,
                     'location': str(loc[0]) + ',' + str(loc[1])}
        res = requests.get('https://restapi.amap.com/v3/geocode/regeo', params=parameter)
        data = res.json()['regeocode']
        sub_data = data['addressComponent']
        data = pd.DataFrame({'address': [data['formatted_address']],
                             'province': [sub_data['province']],
                             'city': [sub_data['city']],
                             'district': [sub_data['district']],
                             'township': [sub_data['township']],
                             'adcode': [sub_data['adcode']],
                             'citycode': [sub_data['citycode']]})
        return data

    def name2loc(self, address: str, city=None):
        parameter = {'key': self.key,
                     'address': address,
                     'city': city}
        res = requests.get('https://restapi.amap.com/v3/geocode/geo', params=parameter)
        data = res.json()['geocodes']
        data_lst = []
        for i in data:
            tmp = pd.DataFrame({'address': [i['formatted_address']],
                                'level': [i['level']],
                                'province': [i['province']],
                                'city': [i['city']],
                                'district': [i['district']],
                                'street': [i['street']],
                                'number': [i['number']],
                                'citycode': [i['citycode']],
                                'adcode': [i['adcode']],
                                'lon': [i['location'].split(',')[0]],
                                'lat': [i['location'].split(',')[1]]})
            data_lst.append(tmp)
        data = pd.concat(data_lst)
        return data


def id_search(kw, mode, type_file, ad_file, key_file):
    pass


if __name__ == '__main__':
    am = AMap('9bcb17cf863e6b02511fb0e06be627e5')
    print(am.name2loc('嘉定'))
