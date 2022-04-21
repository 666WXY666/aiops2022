'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangChengsen
Date: 2022-04-21 22:40:10
LastEditors: WangXingyu
LastEditTime: 2022-04-21 22:44:14
'''
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

nodes = ['node-1', 'node-2', 'node-3', 'node-4', 'node-5', 'node-6']
services = ['adservice-grpc', 'adservice-http', 'cartservice-grpc', 'checkoutservice-grpc', 'currencyservice-grpc',
            'emailservice-grpc', 'frontend-http', 'paymentservice-grpc', 'productcatalogservice-grpc',
            'recommendationservice-grpc', 'shippingservice-grpc']
pods = ['adservice-0', 'adservice-1', 'adservice-2', 'adservice2-0', 'cartservice-0', 'cartservice-1', 'cartservice-2',
        'cartservice2-0', 'checkoutservice-0', 'checkoutservice-2', 'checkoutservice-1', 'checkoutservice2-0',
        'currencyservice-0', 'currencyservice-1', 'currencyservice-2', 'currencyservice2-0', 'emailservice-0',
        'emailservice-1', 'emailservice-2', 'emailservice2-0', 'frontend-0', 'frontend-1', 'frontend-2', 'frontend2-0',
        'paymentservice-0', 'paymentservice-1', 'paymentservice-2', 'paymentservice2-0', 'productcatalogservice-0',
        'productcatalogservice-1', 'productcatalogservice-2', 'productcatalogservice2-0', 'recommendationservice-0',
        'recommendationservice-1', 'recommendationservice-2', 'recommendationservice2-0', 'shippingservice-0',
        'shippingservice-1', 'shippingservice-2', 'shippingservice2-0']

node_kpis = ['cpu.pct_usage', 'cpu.system', 'cpu.user', 'disk.free', 'disk.pct_usage', 'disk.total', 'disk.used',
             'fs.inodes.free', 'fs.inodes.in_use', 'fs.inodes.total', 'fs.inodes.used', 'io.avg_q_sz', 'io.await',
             'io.r_await', 'io.r_s', 'io.rkb_s', 'io.util', 'io.w_await', 'io.w_s', 'load.1', 'load.15', 'load.5',
             'mem.free', 'mem.real.used', 'mem.usable', 'mem.used']
service_kpis = ['mrt']
pod_kpis = ['cpu_cfs_periods', 'cpu_cfs_throttled_periods', 'cpu_cfs_throttled_seconds', 'cpu_system_seconds',
            'cpu_usage_seconds', 'cpu_user_seconds', 'fs_inodes', 'fs_reads', 'fs_reads_MB', 'fs_usage_MB', 'fs_writes',
            'fs_writes_MB', 'memory_cache', 'memory_failures.container.pgfault', 'memory_failures.container.pgmajfault',
            'memory_failures.hierarchy.pgfault', 'memory_failures.hierarchy.pgmajfault', 'memory_mapped_file',
            'memory_rss', 'memory_usage_MB', 'memory_working_set_MB', 'network_receive_MB.eth0',
            'network_receive_packets.eth0', 'network_transmit_MB.eth0', 'network_transmit_packets.eth0', 'threads']


def get_raw_data():
    """
    从原始数据中读取固定格式的df
    :return: n * 1207 (1207 = 6*26 + 11 + 40*26)
    """
    df_node = pd.read_csv(
        '../../data/training_data_normal/cloudbed-1/metric/node/kpi_cloudbed1_metric_0319.csv')
    df_node['node_kpi'] = df_node.apply(
        lambda x: x['cmdb_id'] + ':' + x['kpi_name'], axis=1)
    df_node.drop(['cmdb_id', 'kpi_name'], axis=1, inplace=True)
    df_node = df_node.groupby(['timestamp', 'node_kpi'], as_index=False).mean()
    df_node = pd.pivot(df_node, index='timestamp', columns='node_kpi')
    df_node.columns = [col[1] for col in df_node.columns]
    features_node = [node + ':system.' +
                     kpi for kpi in node_kpis for node in nodes]
    df_node = df_node[features_node].sort_index()

    df_service = pd.read_csv(
        '../../data/training_data_normal/cloudbed-1/metric/service/metric_service.csv')
    df_service['service'] = df_service['service'].apply(lambda x: x + ':mrt')
    df_service.drop(['rr', 'sr', 'count'], axis=1, inplace=True)
    df_service = df_service.groupby(
        ['timestamp', 'service'], as_index=False).mean()
    df_service = pd.pivot(df_service, index='timestamp', columns='service')
    df_service.columns = [col[1] for col in df_service.columns]
    features_service = [service + ':' +
                        kpi for kpi in service_kpis for service in services]
    df_service = df_service[features_service].sort_index()

    dfs_pod = []
    for kpi in pod_kpis:
        df_pod = pd.read_csv(
            '../../data/training_data_normal/cloudbed-1/metric/container/kpi_container_' + kpi.split('.')[0] + '.csv')
        df_pod['cmdb_id'] = df_pod['cmdb_id'].apply(lambda x: x.split('.')[-1])
        df_pod['kpi_name'] = df_pod['kpi_name'].apply(
            lambda x: x.split('./')[0])
        df_pod['pod_kpi'] = df_pod.apply(
            lambda x: x['cmdb_id'] + ':' + x['kpi_name'], axis=1)
        df_pod.drop(['cmdb_id', 'kpi_name'], axis=1, inplace=True)
        df_pod = df_pod.groupby(
            ['timestamp', 'pod_kpi'], as_index=False).mean()

        df_pod = pd.pivot(df_pod, index='timestamp', columns='pod_kpi')
        df_pod.columns = [col[1] for col in df_pod.columns]
        feature_pod = [pod + ':container_' + kpi for pod in pods]
        df_pod = df_pod[feature_pod].sort_index()
        dfs_pod.append(df_pod)
    df_pod = pd.concat(dfs_pod, axis=1)

    df = pd.concat([df_node, df_service, df_pod], axis=1)
    return df


def process_data(df, cmdbs, mode='mean', train=True, scaler_path='../../model/scaler.pkl'):
    """
    返回聚合后的数据
    :param df: n * 1207 (1207 = 6*26 + 11 + 40*26)
    :param cmdbs: nodes名称 + services名称 + pods名称，用于固定返回df的顺序
    :param mode: "mean" or "max"
    :param train: True or Fasle
    :param scaler_path: scaler存放的路径
    :return: n * 57 (57 = 6 + 11 + 40)
    """
    if train:
        scaler = StandardScaler()
        df.iloc[:, :] = np.abs(scaler.fit_transform(df.values))
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df.iloc[:, :] = np.abs(scaler.transform(df.values))

    cmdb2kpi = defaultdict(list)
    for col in df.columns.values:
        cmdb2kpi[col.split(':')[0]].append(col)

    df_new = pd.DataFrame()
    if mode == 'mean':
        for cmdb, kpi in cmdb2kpi.items():
            df_new[cmdb] = df[kpi].apply(lambda x: x.mean(), axis=1)
    elif mode == 'max':
        for cmdb, kpi in cmdb2kpi.items():
            df_new[cmdb] = df[kpi].apply(lambda x: x.max(), axis=1)
    else:
        print('mode must be "mean" or "max".')

    df_new = df_new[cmdbs]
    return df_new


df = get_raw_data()
print(df.shape)
cmdb = nodes + services + pods
df = process_data(df, cmdb)
print(df.shape)
print(df.head())
