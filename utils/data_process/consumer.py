'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-21 12:23:08
LastEditors: WangXingyu
LastEditTime: 2022-04-25 20:04:33
'''
import json
import time
import traceback
from collections import OrderedDict
from threading import Thread

import pandas as pd
import schedule
from kafka import KafkaConsumer

ips = ['10.3.2.41', '10.3.2.4', '10.3.2.36']
kpi_d = OrderedDict()
metric_d = OrderedDict()
trace_d = OrderedDict()
log_d = OrderedDict()


def kpi():
    """
    consume kpi
    """
    while True:
        try:
            my_consumer = KafkaConsumer('kpi-1c9e9efe6847bc4723abd3640527cbe9', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data = json.loads(data)
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - data['timestamp'] % 60
                if kpi_d.get(t) is None:
                    kpi_d[t] = []
                try:
                    kpi_d[t].append(
                        [data['timestamp'], data['cmdb_id'], data['kpi_name'], data['value']])
                except Exception:
                    kpi_d[t].append([data.get('timestamp'), data.get(
                        'cmdb_id'), data.get('kpi_name'), data.get('value')])
                    print(traceback.format_exc())

        except Exception:
            print(traceback.format_exc())


def metric():
    """
    consume metric
    """
    while True:
        try:
            my_consumer = KafkaConsumer('metric-1c9e9efe6847bc4723abd3640527cbe9', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data = json.loads(data)
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - data['timestamp'] % 60
                if metric_d.get(t) is None:
                    metric_d[t] = []
                try:
                    metric_d[t].append(
                        [data['service'], data['timestamp'], data['rr'], data['sr'], data['mrt'], data['count']])
                except Exception:
                    metric_d[t].append([data.get('service'), data.get('timestamp'), data.get(
                        'rr'), data.get('sr'), data.get('mrt'), data.get('count')])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def trace():
    """
    consume trace
    """
    while True:
        try:
            my_consumer = KafkaConsumer('trace-1c9e9efe6847bc4723abd3640527cbe9', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data = json.loads(data)
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp']//1000 - data['timestamp']//1000 % 60
                if trace_d.get(t) is None:
                    trace_d[t] = []
                try:
                    trace_d[t].append([data['timestamp'], data['cmdb_id'], data['span_id'], data['trace_id'],
                                      data['duration'], data['type'], data['status_code'], data['operation_name'], data['parent_span']])
                except Exception:
                    trace_d[t].append([data['timestamp'], data['cmdb_id'], data['span_id'], data['trace_id'],
                                      data['duration'], data['type'], data['status_code'], data['operation_name'], data['parent_span']])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def log():
    """
    consume log
    """
    while True:
        try:
            my_consumer = KafkaConsumer('log-1c9e9efe6847bc4723abd3640527cbe9', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data = json.loads(data)
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - data['timestamp'] % 60
                if log_d.get(t) is None:
                    log_d[t] = []
                try:
                    log_d[t].append([data['log_id'], data['timestamp'], data['cmdb_id'], data['log_name'],
                                     data['value']])
                except Exception:
                    log_d[t].append([data.get('log_id'), data.get('timestamp'), data.get('cmdb_id'),
                                     data.get('log_name'), data.get('value')])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def clean(no):
    """
    clean global topic data dict
    @param no: global topic data dict's len
    """
    while True:
        for d in [kpi_d, metric_d, trace_d, log_d]:
            while len(d) > no:
                try:
                    d.popitem(last=False)
                except Exception:
                    pass
        time.sleep(30)


def test():
    """
    test
    """
    current_time = int(time.time())
    print(time.strftime('%H:%M:%S', time.localtime(current_time)))
    current_time = current_time - current_time % 60
    print(current_time)

    kpi_list = kpi_d.get(current_time - 60, [])
    kpi_df = pd.DataFrame(
        kpi_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])

    metric_list = metric_d.get(current_time - 60, [])
    metric_df = pd.DataFrame(metric_list, columns=[
                             'service', 'timestamp', 'rr', 'sr', 'count', 'mrt'])

    trace_list = trace_d.get(current_time - 60, [])
    trace_df = pd.DataFrame(trace_list, columns=[
                            'timestamp', 'cmdb_id', 'span_id', 'trace_id', 'duration', 'type', 'status_code', 'operation_name', 'parent_span'])

    log_list = log_d.get(current_time - 60, [])
    log_df = pd.DataFrame(
        log_list, columns=['log_id', 'timestamp', 'cmdb_id', 'log_name', 'value'])

    if not (kpi_df.empty or metric_df.empty or trace_df.empty or log_df.empty):
        print('kpi:\n', kpi_df.drop_duplicates(['timestamp'])['timestamp'])
        kpi_df['new'] = kpi_df['kpi_name']
        kpi_df['new'] = kpi_df['new'].apply(
            lambda x: 'network' if 'network' in x else x)
        kpi_df = kpi_df[kpi_df['new'] == 'network']
        print(kpi_df)
        print('metric:\n', metric_df.drop_duplicates(
            ['timestamp'])['timestamp'])
        trace_df.drop_duplicates(['timestamp'], inplace=True)
        trace_df.sort_values(by='timestamp', inplace=True)
        trace_df.reset_index(drop=True, inplace=True)
        print('trace:\n', trace_df['timestamp'].iloc[0] //
              1000, trace_df['timestamp'].iloc[-1]//1000)
        log_df.drop_duplicates(['timestamp'], inplace=True)
        log_df.sort_values(by='timestamp', inplace=True)
        log_df.reset_index(drop=True, inplace=True)
        print('log:\n', log_df['timestamp'].iloc[0],
              log_df['timestamp'].iloc[-1])
        print('\n')


def data_deal():
    Thread(target=kpi).start()
    Thread(target=metric).start()
    # Thread(target=trace).start()
    # Thread(target=log).start()
    Thread(target=clean, args=[100]).start()
    # time.sleep(60)  # 冷启动时间


if __name__ == '__main__':
    data_deal()
    schedule.every().minute.at(':59').do(test)
    while True:
        schedule.run_pending()
        time.sleep(1)
