"""
@File : vec.py
@Author : 杨与桦
@Time : 2023/10/03 16:48
"""
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from traffictools.graph import geo


def coord_trans(
        s: Union[shapely.Point, shapely.LineString, shapely.Polygon, gpd.GeoDataFrame, pd.DataFrame, list, tuple, np.ndarray]
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

    # df附加功能
    df_col = []
    if type(s) == pd.DataFrame:
        df_col = s.columns
        s = s.values

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

    # df附加功能
    if len(df_col) != 0:
        res = pd.DataFrame(data=res, columns=df_col)

    return res


def distance(data: pd.DataFrame, lon_col: list, lat_col: list, r=6371):
    data2 = data.copy()
    data2[lon_col+lat_col] = np.pi*data2[lon_col+lat_col]/180  # 转为弧度制
    dlon = data2[lon_col[1]] - data2[lon_col[0]]
    dlat = data2[lat_col[1]] - data2[lat_col[0]]
    a = np.sin(dlat/2.0)**2 + np.cos(data2[lat_col[0]]) * np.cos(data2[lat_col[1]]) * np.sin(dlon/2.0)**2
    dist = 2 * r * 1000 * np.arcsin(np.sqrt(a))
    return dist


def grid_generate(bounds: list, length=500, r=6371000):
    lon_range = [bounds[0], bounds[2]]
    lat_range = [bounds[1], bounds[3]]
    clat = np.radians(sum(lat_range) / 2)  # 选区的中心纬度，用以近似计算纬线长
    C = 2 * r * np.pi  # 地球周长
    delta_lon = np.round(360 / (np.cos(clat) * C) * length, 6)  # 每栅格跨经度
    delta_lat = np.round(360 / C * length, 6)  # 每栅格跨纬度

    lon_num = np.ceil((lon_range[1] - lon_range[0]) / delta_lon)
    lat_num = np.ceil((lat_range[1] - lat_range[0]) / delta_lat)  # 计算行列号

    data = pd.DataFrame()
    lon_col = []  # 经度编号
    lat_col = []  # 纬度编号
    mid_lon = []  # 中点经度
    mid_lat = []  # 中点纬度
    geometry_list = []  # 几何要素

    for i in range(int(lon_num)):
        for j in range(int(lat_num)):
            # 中心点坐标
            mlon = i * delta_lon + lon_range[0]
            mlat = j * delta_lat + lat_range[0]
            mlon1 = (i + 1) * delta_lon + lon_range[0]
            mlat1 = (j + 1) * delta_lat + lat_range[0]
            # 生成Polygon元素
            grid_ij = shapely.Polygon([
                (mlon - delta_lon / 2, mlat - delta_lat / 2),
                (mlon1 - delta_lon / 2, mlat - delta_lat / 2),
                (mlon1 - delta_lon / 2, mlat1 - delta_lat / 2),
                (mlon - delta_lon / 2, mlat1 - delta_lat / 2)
            ])
            # 存储生成的数据
            lon_col.append(i)
            lat_col.append(j)
            mid_lon.append(mlon)
            mid_lat.append(mlat)
            geometry_list.append(grid_ij)

    data['lon col'] = lon_col
    data['lat col'] = lat_col
    data['mid lon'] = mid_lon
    data['mid lat'] = mid_lat
    data['geometry'] = geometry_list
    grid = gpd.GeoDataFrame(data)

    return grid


def here(ax, how: list[str], draw=True, p=0.01, n=10, alpha=0.3, color='r', lw=1, style='short'):
    dy = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx = ax.get_xlim()[1] - ax.get_xlim()[0]  # 获取图的尺寸
    cy = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    cx = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2  # 获取图的中点，作为初始定位点

    action_label = ['U', 'u', 'D', 'd', 'R', 'r', 'L', 'l']
    action_map = p * np.array([n, 1, -n, -1] * 2)  # 不同指令对应的位移距离（比例）

    locs = []  # 坐标集
    for i in how:
        action_count = np.array(list(map(lambda h: i.count(h), action_label)))  # 统计how中各指令的数量
        action = action_count * action_map * np.array([dy] * 4 + [dx] * 4)  # 指令转换为移动距离
        x = cx + np.sum(action[4:])
        y = cy + np.sum(action[:4])
        locs.append((x, y))

    if draw:
        for i in ax.get_children():  # 修改各要素不透明度
            i.set_alpha(alpha)

        for i in range(len(how)):
            loc = locs[i]
            if style == 'long':
                ax.axhline(y=loc[1], color=color, lw=lw)
                ax.axvline(x=loc[0], color=color, lw=lw)
            elif style == 'short':
                ax.hlines(y=loc[1], xmin=loc[0] - dx / 50, xmax=loc[0] + dx / 50, color=color, lw=lw)
                ax.vlines(x=loc[0], ymin=loc[1] - dx / 50, ymax=loc[1] + dx / 50, color=color, lw=lw)
            ax.text(loc[0], loc[1], i)
    return locs


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


if __name__ == '__main__':
    pass
