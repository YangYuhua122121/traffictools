"""
@File : vec.py
@Author : 杨与桦
@Time : 2023/10/03 16:48
"""
import geopandas as gpd
import shapely
import pandas as pd
from numpy import linspace
from typing import Union
from numpy import array
from geo import CoordTrans


def line_split(line: Union[pd.DataFrame, shapely.LineString], nodes: pd.DataFrame, lon, lat, s=10):
    """
将完整的线对象截断为多个线对象，截断将基于输入的节点
该函数将返回以各点为起点的多条截断线段
    :param s: 截断后各线段的平滑度
    :param lat: 输入数据集中纬度对应的字段名
    :param lon: 输入数据集中经度对应的字段名
    :param line: 一个散点集，组成了轨迹线，包含许多冗余的点;或者一个LineString线矢量
    :param nodes: 一个散点集，仅包含感兴趣的节点（将在此处截断）
    """
    # 获取线
    if isinstance(line, pd.DataFrame):
        line_geometry = shapely.LineString(line[[lon, lat]].values)
    else:
        line_geometry = line
    # 获取线的起终点
    st = pd.DataFrame({'geometry': [line_geometry.interpolate(0)]})
    ed = pd.DataFrame({'geometry': [line_geometry.interpolate(line_geometry.length)]})
    # 构造路段
    link_df = nodes.copy()
    link_df['geometry'] = gpd.points_from_xy(link_df[lon], link_df[lat])
    link_df = pd.concat([st, link_df, ed])
    link_df['mark'] = range(len(link_df))
    link_df['o'] = link_df['geometry'].apply(lambda r: line_geometry.project(r))
    link_df.sort_values(by='o', inplace=True)  # 避免输入分割点乱序造成的分割结果混乱
    mark = (link_df['mark'][:-1]-1).to_frame().reset_index(drop=True)  # 标记分割点的索引
    link_df['geometry1'] = link_df['geometry'].shift(-1)
    link_df = link_df.iloc[:-1]
    link_df['d'] = link_df['geometry1'].apply(lambda r: line_geometry.project(r))

    def get_line(x):
        ls = []
        tmp = linspace(int(x['o']), int(x['d']), s)

        for i in tmp:
            ls.append(line_geometry.interpolate(i))
        return shapely.LineString(ls)

    split_lines = link_df.apply(get_line, axis=1).reset_index(drop=True).to_frame()
    split_lines['p1'] = mark
    split_lines['p2'] = split_lines['p1'].shift(-1)
    split_lines.loc[split_lines.index[-1], 'p2'] = -1
    split_lines['p2'] = split_lines['p2'].astype(int)
    split_lines.rename(columns={0: 'geometry'}, inplace=True)
    return split_lines


def undirected2(df: pd.DataFrame, link: list[any, any]) -> pd.DataFrame:
    df1 = df[[link[0]]+[link[1]]+list(df.columns.drop(link))].copy()
    df1['mark1'] = range(len(df))
    df2 = df[[link[1]]+[link[0]]+list(df.columns.drop(link))].copy()
    df2.rename(columns={link[0]: link[1], link[1]: link[0]}, inplace=True)
    df2['mark1'] = range(len(df))
    df = pd.concat([df1, df2])
    lst = []
    for i in df.groupby(link):
        i[1]['mark1'] = i[1]['mark1'].min()
        i[1]['mark2'] = range(len(i[1]))
        lst.append(i[1])
    res = pd.concat(lst)
    undi_df = res.drop_duplicates(subset=['mark1', 'mark2']).drop(columns=['mark1', 'mark2'])
    return undi_df


def undirected(df: pd.DataFrame, link: list[any, any]) -> pd.DataFrame:
    def historic(n):
        def prime(ii, primess):
            for primee in primess:
                if not (ii == primee or ii % primee):
                    return False
            primess.add(ii)
            return ii
        primes = {2}
        i, pr = 2, 0
        while True:
            if prime(i, primes):
                pr += 1
                if pr == n:
                    return primes
            i += 1
    p1 = list(df[link[0]].unique())
    p2 = list(df[link[1]].unique())
    p = list(set(p1+p2))
    p1_mark = pd.DataFrame({link[0]: p, 'mark1': list(historic(len(p)))})
    p2_mark = pd.DataFrame({link[1]: p, 'mark2': list(historic(len(p)))})
    df = pd.merge(df, p1_mark, on=link[0])
    df = pd.merge(df, p2_mark, on=link[1])
    df['mark'] = df['mark1']*df['mark2']
    new_link = df.drop_duplicates(subset='mark')[link+['mark']]
    df = pd.merge(df, new_link, on='mark')
    df = df.drop(columns=[str(link[0])+'_x', str(link[1])+'_x', 'mark1', 'mark2', 'mark'])
    df.rename(columns={str(link[0])+'_y': link[0], str(link[1])+'_y': link[1]}, inplace=True)
    return df


def s2m(s: Union[shapely.Point, shapely.LineString, shapely.Polygon, None] = None,
        gdf: Union[gpd.GeoDataFrame, None] = None):
    def nc_get(longitude, latitude):
        ct = CoordTrans(lon=longitude, lat=latitude)
        new_lon, new_lat = ct.wgs842gcj02()
        return list(zip(new_lon, new_lat))

    def doit(shape):
        if isinstance(shape, shapely.Polygon):
            lon = array(list(shape.exterior.coords.xy[0]))
            lat = array(list(shape.exterior.coords.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.Polygon(new_coord)
        elif isinstance(shape, shapely.LineString):
            lon = array(list(shape.xy[0]))
            lat = array(list(shape.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.LineString(new_coord)
        else:
            lon = array(list(shape.xy[0]))
            lat = array(list(shape.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.Point(new_coord)

    if not isinstance(s, type(None)):
        res = doit(s)
    else:
        gdf['geometry'] = gdf['geometry'].apply(lambda r: doit(r))
        res = gdf.copy()
    return res


if __name__ == '__main__':
    jd = gpd.read_file('C:/Users/杨/Desktop/交通数据分析实训/2/data/map/network from graph/Jiading_edges.shp')
    print(s2m(gdf=jd))