'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-06 20:50:54
LastEditors: WangXingyu
LastEditTime: 2022-04-24 16:54:42
'''
import os
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import schedule

from algorithm.anomaly_catboost import CatBoost
from algorithm.micro_rca import PageRCA
from algorithm.spot import SPOT
from utils.data_process.consumer import data_deal, kpi_d, metric_d
from utils.data_process.process_data import (get_raw_data, istio_kpis,
                                             node_kpis, nodes, pod_kpis, pods,
                                             process_data, rca_kpis,
                                             service_kpis, services, upsample)
from utils.submit import submit

is_anomaly = {i: 0 for i in nodes+pods+services}

WAIT_FLAG = False

fault_count = 0

type2id = {
    'k8s容器cpu负载': 0,
    'k8s容器内存负载': 1,
    'k8s容器写io负载': 2,
    'k8s容器网络丢包': 3,
    'k8s容器网络延迟': 4,
    'k8s容器网络资源包损坏': 5,
    'k8s容器网络资源包重复发送': 6,
    'k8s容器读io负载': 7,
    'k8s容器进程中止': 8,
    'node 内存消耗': 9,
    'node 磁盘写IO消耗': 10,
    'node 磁盘空间消耗': 11,
    'node 磁盘读IO消耗': 12,
    'node节点CPU故障': 13,
    'node节点CPU爬升': 14
}

id2type = {
    0: 'k8s容器cpu负载',
    1: 'k8s容器内存负载',
    2: 'k8s容器写io负载',
    3: 'k8s容器网络丢包',
    4: 'k8s容器网络延迟',
    5: 'k8s容器网络资源包损坏',
    6: 'k8s容器网络资源包重复发送',
    7: 'k8s容器读io负载',
    8: 'k8s容器进程中止',
    9: 'node 内存消耗',
    10: 'node 磁盘写IO消耗',
    11: 'node 磁盘空间消耗',
    12: 'node 磁盘读IO消耗',
    13: 'node节点CPU故障',
    14: 'node节点CPU爬升'
}


def main(train=True, type='online', run_i=0):
    if train:
        df_node = pd.concat([pd.read_csv('./data/training_data_normal/cloudbed-1/metric/node/kpi_cloudbed1_metric_0319.csv'),
                             pd.read_csv(
                                 './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/node/kpi_cloudbed1_metric_0320.csv'),
                             pd.read_csv('./data/training_data_with_faults/tar/2022-03-21-cloudbed1/metric/node/kpi_cloudbed1_metric_0321.csv')])
        df_node.reset_index(drop=True, inplace=True)

        df_service = pd.concat([pd.read_csv('./data/training_data_normal/cloudbed-1/metric/service/metric_service.csv'),
                                pd.read_csv(
                                    './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/service/metric_service.csv'),
                                pd.read_csv('./data/training_data_with_faults/tar/2022-03-21-cloudbed1/metric/service/metric_service.csv')])
        df_service.reset_index(drop=True, inplace=True)

        dfs_pod = []
        for kpi in pod_kpis:
            df_pod1 = pd.read_csv(
                './data/training_data_normal/cloudbed-1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
            dfs_pod.append(df_pod1)

            df_pod2 = pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
            dfs_pod.append(df_pod2)

            df_pod3 = pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-21-cloudbed1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
            dfs_pod.append(df_pod3)

        df_pod = pd.concat(dfs_pod, ignore_index=True)
        df_pod.reset_index(drop=True, inplace=True)
        df_pod['kpi_name'] = df_pod['kpi_name'].apply(
            lambda x: x.split('./')[0])

        print(df_node, df_service, df_pod)
    else:
        if type == 'online':
            global WAIT_FLAG
            current_time = int(time.time())
            print(time.strftime('%H:%M:%S', time.localtime(current_time)))
            current_time = current_time - current_time % 60
            print(current_time)

            kpi_list = kpi_d.get(current_time - 60, [])
            df_kpi = pd.DataFrame(
                kpi_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])
            print('df_kpi:\n', df_kpi)

            for i in reversed(range(1, 11)):
                df_kpi_10min_list = kpi_d.get(current_time - i*60, [])
            df_kpi_10min = pd.DataFrame(
                df_kpi_10min_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])

            metric_list = metric_d.get(current_time - 60, [])
            df_service = pd.DataFrame(metric_list, columns=[
                'service', 'timestamp', 'rr', 'sr', 'count', 'mrt'])
            print('df_service:\n', df_service)
        else:
            current_time = 1647790210 - 1647790210 % 60
            current_time += run_i*60
            df_service = pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/service/metric_service.csv')
            df_service = df_service[df_service['timestamp'] == current_time]
            df_service.reset_index(drop=True, inplace=True)
            print('service:\n', df_service)

            df_node = pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/node/kpi_cloudbed1_metric_0320.csv')
            dfs_pod = []
            for kpi in pod_kpis:
                df_pod = pd.read_csv(
                    './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/container/kpi_' + kpi.split('.')[0] + '.csv')
                dfs_pod.append(df_pod)
            dfs_pod.append(pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/istio/kpi_istio_request_duration_milliseconds.csv'))
            dfs_pod.append(pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/istio/kpi_istio_agent_go_goroutines.csv'))
            dfs_pod.append(pd.read_csv(
                './data/training_data_with_faults/tar/2022-03-20-cloudbed1/metric/jvm/kpi_java_lang_ClassLoading_TotalLoadedClassCount.csv'))
            df_pod = pd.concat(dfs_pod, ignore_index=True)
            df_kpi = pd.concat([df_node, df_pod], ignore_index=True)

            df_kpi_10min = df_kpi[(df_kpi['timestamp'] >= current_time-9*60) & (
                df_kpi['timestamp'] <= current_time)].reset_index(drop=True)
            print('kpi_10min:\n', df_kpi_10min)

            df_kpi = df_kpi[df_kpi['timestamp'] ==
                            current_time].reset_index(drop=True)
            print('kpi:\n', df_kpi)

        if not (df_kpi.empty or df_kpi_10min.empty):
            df_kpi['kpi_name'] = df_kpi['kpi_name'].apply(
                lambda x: x.split('./')[0])
            df_kpi['value'] = df_kpi['value'].astype('float')
            df_node = df_kpi[df_kpi['kpi_name'].isin(
                node_kpis)].reset_index(drop=True)
            df_pod = df_kpi[df_kpi['kpi_name'].isin(
                pod_kpis)].reset_index(drop=True)

            print('df_node:\n', df_node)
            print('df_pod:\n', df_pod)

            df_rca = df_kpi_10min[df_kpi_10min['kpi_name'].isin(
                rca_kpis)].reset_index(drop=True)
            df_kpi_10min['kpi_name'] = df_kpi_10min['kpi_name'].apply(
                lambda x: 'istio_request_duration_milliseconds' if 'istio_request_duration_milliseconds' in x else x)
            df_istio = df_kpi_10min[df_kpi_10min['kpi_name'].isin(
                istio_kpis)].reset_index(drop=True)
            print('df_rca:\n', df_rca)
            print('df_istio:\n', df_istio)
        else:
            df_node = pd.DataFrame()
            df_pod = pd.DataFrame()
            df_rca = pd.DataFrame()
            df_istio = pd.DataFrame()

    if train:
        rca_timestamp = []
    else:
        rca_timestamp = df_rca.drop_duplicates(
            ['timestamp'])['timestamp'].to_list()
        print('rca_timestamp:\n', rca_timestamp)

    if train or not(df_node.empty or df_service.empty or df_pod.empty or len(rca_timestamp) < 10):
        df_node = get_raw_data(df_node, type='node', train=train)
        df_service = get_raw_data(df_service, type='service', train=train)
        df_pod = get_raw_data(df_pod, type='pod', train=train)

        df = pd.concat([df_node, df_service, df_pod], axis=1)
        print('df:\n', df)

        cmdb = nodes + services + pods
        node_pod_kpis = node_kpis+pod_kpis
        for i in ['container_fs_writes', 'container_fs_writes_MB', 'container_memory_mapped_file']:
            node_pod_kpis.remove(i)
        df_anomaly, df_cat = process_data(df, cmdb, node_pod_kpis, train=train)
        print('df_anomaly:\n', df_anomaly)
        print('df_cat:\n', df_cat)

        spot = SPOT(1e-3)
        anomaly_catboost = CatBoost()
        if train:
            spot.train(df_anomaly)

            df_cat = df_cat.reset_index()
            df_cat['timestamp'] = pd.to_datetime(
                df_cat['timestamp'], unit='s')

            label = pd.concat([pd.read_csv('./data/training_data_with_faults/groundtruth/groundtruth-k8s-1-2022-03-20.csv'),
                               pd.read_csv(
                './data/training_data_with_faults/groundtruth/groundtruth-k8s-1-2022-03-21.csv')])
            label['failure_type'] = label['failure_type'].apply(
                lambda x: type2id[x])
            label['timestamp'] = (pd.to_datetime(
                label['timestamp'], unit='s') + pd.to_timedelta('30s')).round('min')
            label = label[['timestamp', 'failure_type']]

            cat_data = pd.merge(label, df_cat, on='timestamp', how='left')
            cat_data = pd.concat([cat_data, upsample(cat_data)])
            cat_data = cat_data.sample(frac=1.0).reset_index(drop=True)
            cat_data_x = cat_data.iloc[:, 2:]
            cat_data_y = cat_data['failure_type']

            anomaly_catboost.train(cat_data_x.values, cat_data_y.values)

        else:
            res = spot.detect(df_anomaly)

            for idx, abn in enumerate(res):
                if abn == True:
                    is_anomaly[cmdb[idx]] += 1
                else:
                    is_anomaly[cmdb[idx]] = 0

            fault_flag, anomaly_dict = spot.check_anomaly(is_anomaly)
            print(fault_flag)
            print(anomaly_dict)

            if fault_flag:
                global fault_count
                fault_count += 1
                if fault_count == 2:
                    fault_count = 0
                    rca = PageRCA(ts=current_time-60,
                                  fDict=anomaly_dict, responds=df_istio, metric=df_rca)
                    cmdb_ans = rca.do_rca()
                    print('cmdb_ans:', cmdb_ans)

                    type_ans = anomaly_catboost.test(df_cat.values)
                    type_ans = list(map(lambda x: id2type[x], type_ans))[0]
                    print('type_ans:', type_ans)

                    code = submit([str(cmdb_ans), str(type_ans)])
                    print('return_code:', code)
                    print('current_time:', time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print('fault_time:', time.strftime(
                        '%Y-%m-%d %H:%M:%S', time.localtime(current_time)))

                    WAIT_FLAG = True


if __name__ == '__main__':
    type = 'offline_test'
    if type == 'train':
        main(True)

    elif type == 'online_test':
        data_deal()
        schedule.every().minute.at(':59').do(main, False, 'online')

        while True:
            if WAIT_FLAG:
                WAIT_FLAG = False
                schedule.clear()
                print('wait for 5 minutes...')
                time.sleep(60*5)
                schedule.every().minute.at(':59').do(main, False, 'online')

                fault_count = 0
                for i, _ in is_anomaly.items():
                    is_anomaly[i] = 0

            schedule.run_pending()

    elif type == 'offline_test':
        for i in range(-2, 6):
            main(False, 'offline', i)
            print('\n-----------------------------------------------------------\n')

    else:
        print('error type')
