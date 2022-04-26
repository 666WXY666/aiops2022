'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangChengsen
Date: 2022-04-21 22:40:10
LastEditors: WangXingyu
LastEditTime: 2022-04-26 15:03:46
'''
import os
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

node_kpis = ['system.cpu.pct_usage', 'system.cpu.system', 'system.cpu.user', 'system.disk.free', 'system.disk.pct_usage', 'system.disk.total', 'system.disk.used',
             'system.fs.inodes.free', 'system.fs.inodes.in_use', 'system.fs.inodes.total', 'system.fs.inodes.used', 'system.io.avg_q_sz', 'system.io.await',
             'system.io.r_await', 'system.io.r_s', 'system.io.rkb_s', 'system.io.util', 'system.io.w_await', 'system.io.w_s', 'system.load.1', 'system.load.15', 'system.load.5',
             'system.mem.free', 'system.mem.real.used', 'system.mem.usable', 'system.mem.used']
service_kpis = ['mrt']
pod_kpis = ['container_cpu_cfs_periods', 'container_cpu_cfs_throttled_periods', 'container_cpu_cfs_throttled_seconds', 'container_cpu_system_seconds',
            'container_cpu_usage_seconds', 'container_cpu_user_seconds', 'container_fs_inodes', 'container_fs_reads', 'container_fs_reads_MB', 'container_fs_usage_MB', 'container_fs_writes',
            'container_fs_writes_MB', 'container_memory_cache', 'container_memory_failures.container.pgfault', 'container_memory_failures.container.pgmajfault',
            'container_memory_failures.hierarchy.pgfault', 'container_memory_failures.hierarchy.pgmajfault', 'container_memory_mapped_file',
            'container_memory_rss', 'container_memory_usage_MB', 'container_memory_working_set_MB', 'container_network_receive_MB.eth0',
            'container_network_receive_packets.eth0', 'container_network_transmit_MB.eth0', 'container_network_transmit_packets.eth0', 'container_threads']
istio_kpis = ['istio_request_duration_milliseconds']
rca_kpis = ['container_cpu_usage_seconds', 'container_memory_usage_MB',
            'container_network_transmit_packets.eth0']


def fillna_with_mean(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column] = df[column].fillna(mean_val)
    return df


def upsample(df):
    new = []
    for i in range(15):
        new.append(df[df['failure_type'] == i].sample(
            n=10 - len(df[df['failure_type'] == i]), replace=True).copy())
    new = pd.concat(new)
    return new


def get_raw_data(df, type='node', train=True):
    """
    从原始数据中读取固定格式的df
    :return: n * 1207 (1207 = 6*26 + 11 + 40*26)
    """
    timestamp = int(df['timestamp'][0])
    df = df.copy()

    if type == 'node':
        df['node_kpi'] = df.apply(
            lambda x: x['cmdb_id'] + ':' + x['kpi_name'], axis=1)
        df.drop(['cmdb_id', 'kpi_name'], axis=1, inplace=True)
        df = df.groupby(
            ['timestamp', 'node_kpi'], as_index=False)['value'].mean()
        df.sort_values(by='node_kpi', inplace=True)
        df.reset_index(drop=True, inplace=True)

        features_node = [
            node + ':' + kpi for kpi in node_kpis for node in nodes]
        features_node.sort()

        if not train:
            available_node_kpi = df['node_kpi'].tolist()
            addition_features_node = list(
                set(features_node)-set(available_node_kpi))

            if len(addition_features_node) > 0:
                print('addition_features_node:', addition_features_node)
                addition_features_node = pd.DataFrame(
                    addition_features_node, columns=['node_kpi'])
                df = pd.concat(
                    [df, addition_features_node], ignore_index=True)
                df.sort_values(by='node_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                # mean = joblib.load('./model/scaler/mean.pkl')
                std_scaler = joblib.load(
                    './model/scaler/online_std_scaler.pkl')
                mean = std_scaler.mean_

                df['value'] = df.apply(
                    lambda x: mean[x.name] if pd.isnull(x['value']) else x['value'], axis=1)
                df['timestamp'] = df.apply(
                    lambda x: timestamp if pd.isnull(x['timestamp']) else int(x['timestamp']), axis=1)

        df = pd.pivot(df, index='timestamp', columns='node_kpi')
        df.columns = [col[1] for col in df.columns]
        df = df[features_node].sort_index()
        if train:
            df = fillna_with_mean(df)

    elif type == 'service':
        df['service_kpi'] = df['service'].apply(lambda x: x + ':mrt')
        df.drop(['rr', 'sr', 'count'], axis=1, inplace=True)
        df = df.groupby(
            ['timestamp', 'service_kpi'], as_index=False)['mrt'].mean()
        df.sort_values(by='service_kpi', inplace=True)
        df.reset_index(drop=True, inplace=True)

        features_service = [service + ':' +
                            kpi for kpi in service_kpis for service in services]
        features_service.sort()

        if not train:
            available_service_kpi = df['service_kpi'].tolist()
            addition_features_service = list(
                set(features_service)-set(available_service_kpi))

            if len(addition_features_service) > 0:
                print('addition_features_service:', addition_features_service)
                addition_features_service = pd.DataFrame(
                    addition_features_service, columns=['service_kpi'])
                df = pd.concat(
                    [df, addition_features_service], ignore_index=True)
                df.sort_values(by='service_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                # mean = joblib.load('./model/scaler/mean.pkl')
                std_scaler = joblib.load(
                    './model/scaler/online_std_scaler.pkl')
                mean = std_scaler.mean_

                df['mrt'] = df.apply(
                    lambda x: mean[x.name+6*26] if pd.isnull(x['mrt']) else x['mrt'], axis=1)
                df['timestamp'] = df.apply(
                    lambda x: timestamp if pd.isnull(x['timestamp']) else int(x['timestamp']), axis=1)

        df = pd.pivot(df, index='timestamp', columns='service_kpi')
        df.columns = [col[1] for col in df.columns]
        df = df[features_service].sort_index()
        if train:
            df = fillna_with_mean(df)

    elif type == 'pod':
        df['cmdb_id'] = df['cmdb_id'].apply(
            lambda x: x.split('.')[-1])
        df['pod_kpi'] = df.apply(
            lambda x: x['cmdb_id'] + ':' + x['kpi_name'], axis=1)
        df.drop(['cmdb_id', 'kpi_name'], axis=1, inplace=True)
        t = df.value_counts(['pod_kpi'])
        df = df.groupby(
            ['timestamp', 'pod_kpi'], as_index=False)['value'].mean()
        df.sort_values(by='pod_kpi', inplace=True)
        df.reset_index(drop=True, inplace=True)

        features_pod = [pod + ':' + kpi for kpi in pod_kpis for pod in pods]
        features_pod.sort()

        if not train:
            available_pod_kpi = df['pod_kpi'].tolist()
            addition_features_pod = list(
                set(features_pod)-set(available_pod_kpi))

            if len(addition_features_pod) > 0:
                print('addition_features_pod:', addition_features_pod)
                addition_features_pod = pd.DataFrame(
                    addition_features_pod, columns=['pod_kpi'])
                df = pd.concat(
                    [df, addition_features_pod], ignore_index=True)
                df.sort_values(by='pod_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                # mean = joblib.load('./model/scaler/mean.pkl')
                std_scaler = joblib.load(
                    './model/scaler/online_std_scaler.pkl')
                mean = std_scaler.mean_

                df['value'] = df.apply(
                    lambda x: mean[x.name+6*26+11] if pd.isnull(x['value']) else x['value'], axis=1)
                df['timestamp'] = df.apply(
                    lambda x: timestamp if pd.isnull(x['timestamp']) else int(x['timestamp']), axis=1)

        df = pd.pivot(df, index='timestamp', columns='pod_kpi')
        df.columns = [col[1] for col in df.columns]
        df = df[features_pod].sort_index()
        if train:
            df = fillna_with_mean(df)
    else:
        print('type must be "node" or "service" or "pod".')
    return df


def process_data(df, cmdbs, kpis, mode='mean', type='online_test', path='./model/scaler/'):
    """
    返回聚合后的数据
    :param df: n * 1207 (1207 = 6*26 + 11 + 40*26)
    :param cmdbs: 用于固定返回df_cmdb的顺序
    :param kpis: 用于固定返回df_kpi的顺序
    :param mode: "mean" or "max"
    :param type: "train" or "online_test" or "offline_test"
    :param scaler_path: scaler存放的路径
    :return: df_cmdb n * 57 (57 = 6 + 11 + 40), df_kpi n * 52 (52 = 26 + 26)
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if type == 'train':
        std = np.std(df.iloc[1440:, :].values, axis=0)
        random_nums = []
        for i in range(1207):
            random_nums.append(np.random.normal(0, 0.01*std[i], size=1440))
        random_nums = np.array(random_nums).T
        df.iloc[:1440, :] = df.iloc[:1440, :] + random_nums
        # mean = df.mean().tolist()
        # mean1 = [1 if x == 0 else x for x in mean]
        # joblib.dump(mean, path + 'mean.pkl')
        # df.iloc[:, :] = np.abs((df.values-mean)/mean1)
        std_scaler = StandardScaler()
        df.iloc[:, :] = np.abs(std_scaler.fit_transform(df.values))
        joblib.dump(std_scaler, path + 'offline_std_scaler.pkl')
        joblib.dump(std, path + 'std.pkl')
    elif type == 'offline_test':
        # mean = joblib.load(path + 'mean.pkl')
        # mean1 = [1 if x == 0 else x for x in mean]
        # df.iloc[:, :] = np.abs((df.values-mean)/mean1)
        std_scaler = joblib.load(path + 'offline_std_scaler.pkl')
        df.iloc[:, :] = np.abs(std_scaler.transform(df.values))
    elif type == 'online_test':
        std_scaler = joblib.load(path + 'online_std_scaler.pkl')
        df.iloc[:, :] = np.abs(std_scaler.transform(df.values))

    cmdb2kpi = defaultdict(list)
    kpi2cmdb = defaultdict(list)
    for col in df.columns.values:
        cmdb2kpi[col.split(':')[0]].append(col)
        kpi2cmdb[col.split(':')[1]].append(col)

    df_cmdb = pd.DataFrame()
    df_kpi = pd.DataFrame()
    if mode == 'mean':
        for cmdb, kpi in cmdb2kpi.items():
            df_cmdb[cmdb] = df[kpi].apply(lambda x: x.mean(), axis=1)
        for kpi, cmdb in kpi2cmdb.items():
            df_kpi[kpi] = df[cmdb].apply(lambda x: x.mean(), axis=1)
    elif mode == 'max':
        for cmdb, kpi in cmdb2kpi.items():
            df_cmdb[cmdb] = df[kpi].apply(lambda x: x.max(), axis=1)
        for kpi, cmdb in kpi2cmdb.items():
            df_kpi[kpi] = df[cmdb].apply(lambda x: x.max(), axis=1)
    else:
        print('mode must be "mean" or "max".')

    df_cmdb = df_cmdb[cmdbs]
    df_kpi = df_kpi[kpis]
    return df_cmdb, df_kpi
