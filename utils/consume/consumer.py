'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-21 12:23:08
LastEditors: WangXingyu
LastEditTime: 2022-04-21 12:23:10
'''
import json
import time
import traceback
from collections import OrderedDict
from threading import Thread

import pandas as pd
import requests
from kafka import KafkaConsumer

ips = ['10.3.2.41', '10.3.2.4', '10.3.2.36']
kpi_d = OrderedDict()
metric_d = OrderedDict()
trace_d = OrderedDict()
log_d = OrderedDict()


# def submit(ctx):
#     assert (isinstance(ctx, list))
#     for tp in ctx:
#         assert (isinstance(tp, list))
#         assert (len(tp) == 2)
#         assert (isinstance(tp[0], str))
#         assert (isinstance(tp[1], str) or (tp[1] is None))
#     data = {'content': json.dumps(ctx)}
#     r = requests.post('http://10.3.2.25:5000/standings/submit/', data=json.dumps(data))
#     return r.text


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
            if len(d) > no:
                try:
                    d.popitem(last=False)
                except Exception:
                    pass
        time.sleep(30)


def test(t):
    """
    test
    @param t: test time delay
    """
    while True:
        current_time = int(time.time())
        current_time = current_time - current_time % 60
        print(current_time)

        kpi_list = []
        for i in reversed(range(7)):
            kpi_list += kpi_d.get(current_time - i * 60, [])
        kpi_df = pd.DataFrame(
            kpi_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])
        print('kpi:\n', kpi_df.drop_duplicates(['timestamp'])['timestamp'])
        kpi_df['kpi_new_name'] = kpi_df['kpi_name']
        kpi_df['kpi_new_name'] = kpi_df['kpi_new_name'].apply(
            lambda x: 'istio_request_duration_milliseconds' if 'istio_request_duration_milliseconds' in x else x)
        test = kpi_df[kpi_df['kpi_new_name'] ==
                      'istio_request_duration_milliseconds']
        print(len(test), test)

        metric_list = []
        for i in reversed(range(5)):
            metric_list += metric_d.get(current_time - i * 60, [])
        metric_df = pd.DataFrame(metric_list, columns=[
                                 'service', 'timestamp', 'rr', 'sr', 'count', 'mrt'])
        print('metric:\n', metric_df.drop_duplicates(
            ['timestamp'])['timestamp'])

        trace_list = []
        for i in reversed(range(6)):
            trace_list += trace_d.get(current_time - i * 60, [])
        trace_df = pd.DataFrame(trace_list, columns=[
                                'timestamp', 'cmdb_id', 'span_id', 'trace_id', 'duration', 'type', 'status_code', 'operation_name', 'parent_span'])
        print('trace:\n', trace_df.drop_duplicates(['timestamp'])['timestamp'])

        log_list = []
        for i in reversed(range(7)):
            log_list += log_d.get(current_time - i * 60, [])
        log_df = pd.DataFrame(
            log_list, columns=['log_id', 'timestamp', 'cmdb_id', 'log_name', 'value'])
        print('log:\n', log_df.drop_duplicates(['timestamp'])['timestamp'])

        print('\n')
        time.sleep(t)


def data_deal():
    Thread(target=kpi).start()
    Thread(target=metric).start()
    Thread(target=trace).start()
    Thread(target=log).start()
    Thread(target=clean, args=[30]).start()
    Thread(target=test, args=[5]).start()


if __name__ == '__main__':
    data_deal()
