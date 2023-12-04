"""
@File : vec.py
@Author : 杨与桦
@Time : 2023/10/03 16:48
"""
import geopandas as gpd
import shapely
import pandas as pd
from typing import Union
from traffictools.graph import geo
import numpy as np


def link_get(line: Union[gpd.GeoDataFrame, shapely.LineString],
             nodes: gpd.GeoDataFrame,
             name: Union[str, int, float, None] = None, s=10):
    """
将完整的线对象截断为多个线对象，截断将基于输入的节点
该函数将返回以各点为起点的多条截断线段
    :param name: 点id对应的字段aa
    :param s: 截断后各线段的平滑度
    :param line: 线
    :param nodes: 一个散点集，仅包含感兴趣的节点（将在此处截断）
    """
    # 获取线
    if (isinstance(line, gpd.GeoDataFrame)) and (len(line) != 1):
        line_geometry = shapely.LineString(line['geometry'])
    elif (isinstance(line, gpd.GeoDataFrame)) and (len(line) == 1):
        line_geometry = line['geometry'].values[0]
    else:
        line_geometry = line
    # 获取线的起终点
    st = pd.DataFrame({'geometry': [line_geometry.interpolate(0)]})
    ed = pd.DataFrame({'geometry': [line_geometry.interpolate(line_geometry.length)]})

    link_df = nodes.copy()
    link_df = pd.concat([st, link_df, ed])
    link_df['o'] = link_df['geometry'].apply(lambda r: line_geometry.project(r))
    link_df.sort_values(by='o', inplace=True)  # 将各点但顺序排序

    link_df['geometry1'] = link_df['geometry'].shift(-1)  # 组成路段前后点
    link_df['d'] = link_df['o'].shift(-1)
    if isinstance(name, (str, int, float)):
        link_df[f'_{name}'] = link_df[name].shift(-1)
    link_df = link_df.iloc[:-1]  # 去掉多余点

    def get_line(x):
        ls = []
        tmp = np.linspace(x['o'], x['d'], s)
        for i in tmp:
            ls.append(line_geometry.interpolate(i))
        return shapely.LineString(ls)

    link_df['line_geometry'] = link_df.apply(get_line, axis=1)
    link_df.drop(columns=['o', 'd', 'geometry1', 'geometry'], inplace=True)
    link_df.rename(columns={'line_geometry': 'geometry'}, inplace=True)
    link_df = gpd.GeoDataFrame(link_df).reset_index(drop=True)
    if isinstance(name, (str, int, float)):
        link_df.loc[link_df.index[0], name] = 'start'
        link_df.loc[link_df.index[-1], f'_{name}'] = 'end'
    return link_df


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
        def prime(ii, primes_s):
            for prime_e in primes_s:
                if not (ii == prime_e or ii % prime_e):
                    return False
            primes_s.add(ii)
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


def coord_trans(
        s: Union[shapely.Point, shapely.LineString, shapely.Polygon, gpd.GeoDataFrame, list, tuple, np.ndarray]
                ):
    def nc_get(longitude, latitude):
        ct = geo.CoordTrans(lon=longitude, lat=latitude)
        new_lon, new_lat = ct.wgs842gcj02()
        return list(zip(new_lon, new_lat))

    def doit(shape):
        if isinstance(shape, shapely.Polygon):
            lon = np.array(list(shape.exterior.coords.xy[0]))
            lat = np.array(list(shape.exterior.coords.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.Polygon(new_coord)
        elif isinstance(shape, shapely.LineString):
            lon = np.array(list(shape.xy[0]))
            lat = np.array(list(shape.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.LineString(new_coord)
        else:
            lon = np.array(list(shape.xy[0]))
            lat = np.array(list(shape.xy[1]))
            new_coord = nc_get(lon, lat)
            return shapely.Point(new_coord)

    if isinstance(s, gpd.GeoDataFrame):
        s['geometry'] = s['geometry'].apply(lambda r: doit(r))
        res = s.copy()
    elif isinstance(s, (list, tuple, np.ndarray)):
        a = np.array(s)  # 转ndarray
        if a.ndim == 1:
            res = nc_get(np.array([a[0]]), np.array([a[1]]))
        else:
            res = nc_get(a[:, 0], a[:, 1])
        if isinstance(s, np.ndarray):
            res = np.array(res)
    else:
        res = doit(s)
    return res


def here(ax, how='', draw=True, p=0.01, alpha=0.3, color='#1f77b4', xy=False):
    ax_copy = ax
    dy = ax_copy.get_ylim()[1] - ax_copy.get_ylim()[0]
    dx = ax_copy.get_xlim()[1] - ax_copy.get_xlim()[0]  # 获取图的尺寸
    my = (ax_copy.get_ylim()[1] + ax_copy.get_ylim()[0]) / 2
    mx = (ax_copy.get_xlim()[1] + ax_copy.get_xlim()[0]) / 2  # 获取图的中点，作为初始定位点
    action_label = ['T', 't', 'D', 'd', 'R', 'r', 'L', 'l']
    action_map = p * np.array([10, 1, -10, -1] * 2)  # 不同指令对应的位移距离（比例）
    action_count = np.array(list(map(lambda h: how.count(h), action_label)))  # 统计how中各指令的数量
    action = action_count * action_map * np.array([dy] * 4 + [dx] * 4)  # 指令转换为移动距离
    x = mx + np.sum(action[4:])
    y = my + np.sum(action[:4])

    if draw:
        for i in ax_copy.get_children():
            i.set_alpha(alpha)

        ax_copy.axhline(y=y, color=color)
        ax_copy.axvline(x=x, color=color)
        return ax_copy
    else:
        if xy:  # 若指定平面像素坐标输出
            loc = (x, y)
        else:
            loc = (x, y)
        return loc


if __name__ == '__main__':
    ps = [(1, 2), (2, 5)]
    tran_ps = coord_trans(ps)
