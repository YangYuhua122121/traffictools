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
import matplotlib.pyplot as plt


class AMap:
    def __init__(self, key: str):
        self.key = key

    def poi(self, types: Union[int, str], page: int, mode: str = 'area',
            area: Union[None, int, str] = None,
            loc: Union[None, tuple] = None,
            r: Union[None, int] = None,
            poly: Union[None, list[tuple]] = None):
        flag = True
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
        poi_data.drop(columns=['type'], inplace=True)
        return poi_data, page, flag

    def ad(self, name: Union[int, str]):
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


if __name__ == '__main__':
    from vec import s2m
    fig = plt.figure(1, (8, 12))
    ax = plt.subplot(111)
    am = AMap('b298651ea09a16dba043a23ab23559ce')
    a = am.ad(310114)
    # ss = s2m(a['geometry'][0])
    # a['geometry'] = ss
    jd = gpd.read_file('C:/Users/杨/Desktop/交通数据分析实训/2/data/map/polygon from osm/Jiading_polygon.shp')
    jd['geometry'] = s2m(jd['geometry'][0])
    jd.plot(alpha=0.2, color='r', ax=ax)
    a['geometry'].plot(ax=ax, alpha=0.2)
    plt.show()
