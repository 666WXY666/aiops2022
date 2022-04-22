'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangChengsen
Date: 2022-04-21 22:40:10
LastEditors: WangXingyu
LastEditTime: 2022-04-22 23:26:12
'''
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import schedule
from sklearn.preprocessing import StandardScaler

from consumer import data_deal, kpi_d, metric_d, trace_d

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


def fillna_with_mean(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column] = df[column].fillna(mean_val)
    return df


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
            available_node_kpi = df.drop_duplicates(
                ['node_kpi'])['node_kpi'].tolist()
            available_node_kpi.sort()

            addition_features_node = list(
                set(features_node)-set(available_node_kpi))

            if len(addition_features_node) > 0:
                addition_features_node = pd.DataFrame(
                    addition_features_node, columns=['node_kpi'])
                df = pd.concat(
                    [df, addition_features_node], ignore_index=True)
                df.sort_values(by='node_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                scaler = joblib.load('../../model/scaler.pkl')
                mean = scaler.mean_

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
            available_service_kpi = df.drop_duplicates(
                ['service_kpi'])['service_kpi'].tolist()
            available_service_kpi.sort()

            addition_features_service = list(
                set(features_service)-set(available_service_kpi))

            if len(addition_features_service) > 0:
                addition_features_service = pd.DataFrame(
                    addition_features_service, columns=['service_kpi'])
                df = pd.concat(
                    [df, addition_features_service], ignore_index=True)
                df.sort_values(by='service_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                scaler = joblib.load('../../model/scaler.pkl')
                mean = scaler.mean_

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
        df['kpi_name'] = df['kpi_name'].apply(
            lambda x: x.split('./')[0])
        df['pod_kpi'] = df.apply(
            lambda x: x['cmdb_id'] + ':' + x['kpi_name'], axis=1)
        df.drop(['cmdb_id', 'kpi_name'], axis=1, inplace=True)
        df = df.groupby(
            ['timestamp', 'pod_kpi'], as_index=False)['value'].mean()
        df.sort_values(by='pod_kpi', inplace=True)
        df.reset_index(drop=True, inplace=True)

        features_pod = [pod + ':' + kpi for kpi in pod_kpis for pod in pods]
        features_pod.sort()

        if not train:
            available_pod_kpi = df.drop_duplicates(
                ['pod_kpi'])['pod_kpi'].tolist()
            available_pod_kpi.sort()

            addition_features_pod = list(
                set(features_pod)-set(available_pod_kpi))

            if len(addition_features_pod) > 0:
                addition_features_pod = pd.DataFrame(
                    addition_features_pod, columns=['pod_kpi'])
                df = pd.concat(
                    [df, addition_features_pod], ignore_index=True)
                df.sort_values(by='pod_kpi', inplace=True)
                df.reset_index(drop=True, inplace=True)

                scaler = joblib.load('../../model/scaler.pkl')
                mean = scaler.mean_

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


def main(train=True):
    if train:
        df_node = pd.read_csv(
            '../../data/training_data_normal/cloudbed-1/metric/node/kpi_cloudbed1_metric_0319.csv')

        df_service = pd.read_csv(
            '../../data/training_data_normal/cloudbed-1/metric/service/metric_service.csv')

        dfs_pod = []
        for kpi in pod_kpis:
            df_pod = pd.read_csv(
                '../../data/training_data_normal/cloudbed-1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
            dfs_pod.append(df_pod)
        df_pod = pd.concat(dfs_pod, ignore_index=True)
    else:
        current_time = int(time.time())
        print(time.strftime('%H:%M:%S', time.localtime(current_time)))
        current_time = current_time - current_time % 60
        print(current_time)

        kpi_list = kpi_d.get(current_time - 60, [])
        df_kpi = pd.DataFrame(
            kpi_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])
        print('df_kpi:\n', df_kpi)

        metric_list = metric_d.get(current_time - 60, [])
        df_service = pd.DataFrame(metric_list, columns=[
            'service', 'timestamp', 'rr', 'sr', 'count', 'mrt'])
        print('df_service:\n', df_service)

        trace_list = trace_d.get(current_time - 60, [])
        df_trace = pd.DataFrame(trace_list, columns=[
            'timestamp', 'cmdb_id', 'span_id', 'trace_id', 'duration', 'type', 'status_code', 'operation_name', 'parent_span'])
        print('df_trace:\n', df_trace)

        # df_service = pd.read_csv(
        #     '../../data/training_data_normal/cloudbed-1/metric/service/metric_service.csv')
        # df_service = df_service[df_service['timestamp'] == 1647626400]
        # df_service.reset_index(drop=True, inplace=True)
        # print('service:\n', df_service)

        # df_node = pd.read_csv(
        #     '../../data/training_data_normal/cloudbed-1/metric/node/kpi_cloudbed1_metric_0319.csv')
        # dfs_pod = []
        # for kpi in pod_kpis:
        #     df_pod = pd.read_csv(
        #         '../../data/training_data_normal/cloudbed-1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
        #     dfs_pod.append(df_pod)
        # dfs_pod.append(pd.read_csv(
        #     '../../data/training_data_normal/cloudbed-1/metric/istio/kpi_istio_request_duration_milliseconds.csv'))
        # dfs_pod.append(pd.read_csv(
        #     '../../data/training_data_normal/cloudbed-1/metric/istio/kpi_istio_agent_go_goroutines.csv'))
        # dfs_pod.append(pd.read_csv(
        #     '../../data/training_data_normal/cloudbed-1/metric/jvm/kpi_java_lang_ClassLoading_TotalLoadedClassCount.csv'))
        # df_pod = pd.concat(dfs_pod, ignore_index=True)
        # df_kpi = pd.concat([df_node, df_pod], ignore_index=True)
        # df_kpi = df_kpi[df_kpi['timestamp'] == 1647626400]
        # df_kpi.reset_index(drop=True, inplace=True)
        # print('kpi:\n', df_kpi)

        if not df_kpi.empty:
            df_kpi['kpi_name'] = df_kpi['kpi_name'].apply(
                lambda x: 'istio_request_duration_milliseconds' if 'istio_request_duration_milliseconds' in x else x)
            df_node = df_kpi[df_kpi['kpi_name'].isin(
                node_kpis)].reset_index(drop=True)
            df_pod = df_kpi[df_kpi['kpi_name'].isin(
                pod_kpis)].reset_index(drop=True)
            df_istio = df_kpi[df_kpi['kpi_name'].isin(
                istio_kpis)].reset_index(drop=True)

            print('df_node:\n', df_node)
            print('df_pod:\n', df_pod)

            pod2node = {i[1]: i[0] for i in df_pod.drop_duplicates(['cmdb_id'])['cmdb_id'].apply(
                lambda x: x.split('.')).to_list()}
        else:
            df_node = pd.DataFrame()
            df_pod = pd.DataFrame()
            df_istio = pd.DataFrame()
            pod2node = {}

    if not (df_node.empty or df_service.empty or df_pod.empty):
        df_node = get_raw_data(df_node, type='node', train=train)
        df_service = get_raw_data(df_service, type='service', train=train)
        df_pod = get_raw_data(df_pod, type='pod', train=train)

        df = pd.concat([df_node, df_service, df_pod], axis=1)
        print('df:\n', df)
        cmdb = nodes + services + pods
        df_new = process_data(df, cmdb, mode='mean', train=train)
        print('df_new:\n', df_new)

    if not train:
        print('df_istio:\n', df_istio)
        print('pod2node:\n', pod2node)
        print('df_trace:\n', df_trace)


if __name__ == '__main__':
    train = True
    if train:
        main(True)
    else:
        data_deal()
        schedule.every().minute.at(':59').do(main, False)
        while True:
            schedule.run_pending()
            time.sleep(1)
