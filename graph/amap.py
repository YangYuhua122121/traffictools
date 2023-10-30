"""
@File : amap.py
@Author : 杨与桦
@Time : 2023/10/26 15:54
"""
import requests
import pandas as pd
from typing import Union
from shapely import Polygon
import geopandas as gpd


class AMap:
    def __init__(self, key: str):
        self.key = key

    def poi_search(self, types: list, page: int, mode: str = 'area',
                   area: Union[None, int, str] = None,
                   loc: Union[None, tuple] = None,
                   r: Union[None, int] = None,
                   poly: Union[None, list[tuple]] = None):
            flag = True
            types = '|'.join(types)
            parameters = {'key': self.key,
                          'types': types,
                          'citylimit': 'true',
                          'page': page,
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
            res = requests.get(url=url, params=parameters)
            poi_data = res.json()['pois']
            if len(poi_data) == 0:
                poi_data = pd.DataFrame()
                flag = False
                return poi_data, page, flag
            poi_data = pd.DataFrame(poi_data)[info]
            poi_data['type1'] = poi_data['type'].apply(lambda x: x.split(';')[0])
            poi_data['type2'] = poi_data['type'].apply(lambda x: x.split(';')[1])
            poi_data['type3'] = poi_data['type'].apply(lambda x: x.split(';')[2])
            poi_data['lon'] = poi_data['location'].apply(lambda x: x.split(',')[0])
            poi_data['lat'] = poi_data['location'].apply(lambda x: x.split(',')[1])
            poi_data['geometry'] = gpd.points_from_xy(poi_data['lon'], poi_data['lat'])
            poi_data.drop(columns=['type', 'location'], inplace=True)
            poi_data = gpd.GeoDataFrame(poi_data)
            return poi_data, page, flag

    def region_get(self, name: Union[int, str]):
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

    def route_plan(self, o, d, mode='car'):
        param_lst = ['walk', 'bus', 'car', 'bike']
        url_list = ['walking', 'transit/integrated', 'driving', 'bicycling']
        url_kw = url_list[param_lst.index(mode)]
        parameter = {'key': self.key,
                     'origin': o,
                     'destination': d,
                     }
        res = requests.get(f'https://restapi.amap.com/v3/direction/{url_kw}', params=parameter)
        data = res.json()['route']['paths']
        df = pd.DataFrame(data[0]['steps'])
        df = df[['polyline']]
        df['total_distance'] = data[0]['distance']
        df['total_duration'] = data[0]['duration']

        def str2line(x):
            point_lst = x.split(';')[0]

        df['polyline'] = df['polyline'].apply(str2line)
        print(df)


if __name__ == '__main__':
    am = AMap('9bcb17cf863e6b02511fb0e06be627e5')
    am.route_plan('116.434307,39.90909', '116.434446,39.90816', mode='walk')
