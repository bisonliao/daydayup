# -*- coding: utf-8 -*-
# 对一批推拉流的日志做分析，从地域分布等角度给一个appid画像

import csv
from matplotlib import pyplot as plt
import numpy as np
import json
from sklearn.cluster import KMeans


## 各省的经纬度，暂时没有用起来
province_location = {
    "Shandong": [117.000923, 36.675807],
    "Hebei": [115.48333, 38.03333],
    "Jilin": [125.35000, 43.88333],
    "Heilongjiang": [127.63333, 47.75000],
    "Liaoning": [123.38333, 41.80000],
    "Neimenggu": [111.670801, 41.818311],
    "Xinjiang": [87.68333, 43.76667],
    "Gansu": [103.73333, 36.03333],
    "Ningxia": [106.26667, 37.46667],
    "Shanxi": [112.53333, 37.86667],
    "Shaanxi": [108.95000, 34.26667],
    "Henan": [113.65000, 34.76667],
    "Anhui": [117.283042, 31.86119],
    "Jiangsu": [119.78333, 32.05000],
    "Zhejiang": [120.20000, 30.26667],
    "Fujian": [118.30000, 26.08333],
    "Guangdong": [113.23333, 23.16667],
    "Jiangxi": [115.90000, 28.68333],
    "Hainan": [110.35000, 20.01667],
    "Guangxi": [108.320004, 22.82402],
    "Guizhou": [106.71667, 26.56667],
    "Hunan": [113.00000, 28.21667],
    "Hubei": [114.298572, 30.584355],
    "Sichuan": [104.06667, 30.66667],
    "Yunnan": [102.73333, 25.05000],
    "Xizang": [91.00000, 30.60000],
    "Qinghai": [96.75000, 36.56667],
    "Tianjin": [117.20000, 39.13333],
    "Shanghai": [121.55333, 31.20000],
    "Chongqing": [106.45000, 29.56667],
    "Beijing": [116.41667, 39.91667],
    "Taiwan": [121.30, 25.03],
    "Hongkong": [114.10000, 22.20000],
    "Aomen": [113.50000, 22.20000]
}

## 从csv文件里加载拉流的信息
def load_play_data():
    with open("E:\\work\\data_mining\\play.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        l = [row for row in reader]
    return l

## 从csv文件里加载推流信息
def load_publish_data():
    with open("E:\\work\\data_mining\\publish.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        l = [row for row in reader]
    return l


## 根据网上的边境经纬度信息，画中国地图，被其他函数调用，作为背景图
def draw_china_map():
    cordfile = "E:\\work\\data_mining\\china.json"
    x = list()
    y = list()

    with open(cordfile, "r", encoding="utf-8") as f:
        data = json.load(f) # type:dict
        data = data['features']  #type:list
        data = data[0] # type:dict
        data = data['geometry'] # type:dict
        data = data['coordinates'] # type:list
        for item in data:
            count = 0
            for cord in item[0]:
                count += 1
                #if (count%113) != 1 :
                    #continue
                x.append(cord[0])
                y.append(cord[1])
    plt.scatter(x, y, linewidths=0.1, color='black')

## 画直方图，后来没有用，直接导出数据用excel好操作一点
def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 200)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin, Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin, Ymax)
    plt.title(Title)
    plt.show()

## 每条流的拉流的用户数 的概率分布
def play_num_distribute(play:list):
    stream_info = dict()
    for row in play:
        stream_id = row['stream_id']
        if not (stream_id in stream_info):
            stream_info[stream_id] = set()
        info = stream_info[stream_id]  # type:set
        info.add(row['id_name'])

    distr = dict()
    for stream_id in stream_info.keys():
        playerlist = stream_info[stream_id] # type:set

        low = len(playerlist) // 10 * 10
        key = "%d"%(low+10)
        if not (key in distr):
            distr[key]=[1, len(playerlist)]
        stream_number, player_number = distr[key]
        distr[key] = [stream_number+1, player_number+len(playerlist)]
    for key in distr.keys():
        print(key,",", distr[key][0], ",", distr[key][1])


## 推流用户的省份分布情况
def publish_geo_distribute(publish:list):
    geo_info = dict()
    for row in publish:
        region = row['geoip.region_name']
        if not (region in geo_info):
            geo_info[region] = set()
        geo_info[region].add(row['stream_id'])
    for region in geo_info.keys():
        print(region, ",", len(geo_info[region]))


## 工具函数，用于对经纬度信息做聚类分布，data是一个数组，里面有很多点的经纬度信息
def cluster_analyze(data:np.ndarray):
    estimator = KMeans(n_clusters=4)#构造聚类器
    estimator.fit(data)
    # label_pred = estimator.labels_ #获取聚类标签
    # centroids = estimator.cluster_centers_ #获取聚类中心
    # inertia = estimator.inertia_ # 获取聚类准则的总和
    cluster_id = estimator.predict(data)
    fig = plt.figure()
    draw_china_map()
    plt.scatter(data[:,0], data[:,1], c=cluster_id, cmap='rainbow', linewidths=1)
    plt.show()
    fig.savefig('e:/plot.png')


##  对推流用户的地理位置，聚类分析
def publish_cluster(publish:list):
    pointers = []
    for row in publish:
        if row['geoip.longitude']=='' or row['geoip.latitude']=='' or row['geoip.country_name'] != 'China':
            continue
        pointers.append([row['geoip.longitude'], row['geoip.latitude']])
    data = np.ndarray(shape=(len(pointers), 2), dtype=float)
    index = 0
    for p in pointers:
        data[index][0] = p[0]
        data[index][1] = p[1]
        index += 1
    cluster_analyze(data)


## 对拉流用户的地理位置，做聚类分析
def play_cluster(play:list):
    pointers = []
    for row in play:
        if row['geoip.longitude']=='' or row['geoip.latitude']=='' or row['geoip.country_name'] != 'China':
            continue
        pointers.append([row['geoip.longitude'], row['geoip.latitude']])
    data = np.ndarray(shape=(len(pointers), 2), dtype=float)
    index = 0
    for p in pointers:
        data[index][0] = p[0]
        data[index][1] = p[1]
        index += 1
    cluster_analyze(data)

## 抽查单条流，对其拉流的用户做聚类分析和地理位置展现
def one_stream_player_cluster(play:list):
    stream_info = dict()
    for row in play:
        if row['geoip.longitude']=='' or row['geoip.latitude']=='' or row['geoip.country_name'] != 'China':
            continue
        stream_id = row['stream_id']
        if not (stream_id in stream_info):
            stream_info[stream_id] = list()
        info = stream_info[stream_id]  # type:list
        info.append(row)

    for stream_id in stream_info.keys():
        rowlist = stream_info[stream_id] # type:list

        rownum = len(rowlist)
        if rownum > 50:
            data = np.ndarray(shape=(rownum, 2), dtype=float)
            index = 0
            for row in rowlist:
                data[index][0] = row['geoip.longitude']
                data[index][1] = row['geoip.latitude']
                index += 1
            cluster_analyze(data)



def main():
    play = load_play_data()
    publish = load_publish_data()
    #play_num_distribute(play)
    #publish_geo_distribute(publish)
    #publish_cluster(publish)
    one_stream_player_cluster(play)



main()
