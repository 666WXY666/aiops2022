'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-19 19:33:56
LastEditors: WangXingyu
LastEditTime: 2022-04-19 19:33:59
'''
import json
import time
import traceback
from collections import OrderedDict
from threading import Thread

import requests
from kafka import KafkaConsumer

ips = ['10.3.2.41', '10.3.2.4', '10.3.2.36']
# ips = ['127.0.0.1']
hour_8_sec = 8 * 60 * 60
kpi_d = OrderedDict()
metric_d = OrderedDict()
trace_d = OrderedDict()
log_d = OrderedDict()


def submit(ctx):
    assert (isinstance(ctx, list))
    for tp in ctx:
        assert (isinstance(tp, list))
        assert (len(tp) == 2)
        assert (isinstance(tp[0], str))
        assert (isinstance(tp[1], str) or (tp[1] is None))
    data = {'content': json.dumps(ctx)}
    r = requests.post(
        'http://10.3.2.25:5000/standings/submit/', data=json.dumps(data))
    return r.text


def kpi():
    """
    consume a-kpi
    """
    while True:
        try:
            my_consumer = KafkaConsumer('a-kpi', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                if len(data['timestamp']) == 13:
                    data['timestamp'] = data['timestamp'][0:-3]
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - (data['timestamp'] + hour_8_sec) % 60
                if kpi_d.get(t) is None:
                    kpi_d[t] = []
                try:
                    kpi_d[t].append([data['timestamp'], data['rr'], data['sr'], data['count'], data['mrt'],
                                     data['tc']])
                except Exception:
                    kpi_d[t].append([data.get('timestamp'), data.get('rr'), data.get('sr'), data.get('count'),
                                     data.get('mrt'), data.get('tc')])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def metric():
    """
    consume a-metric
    """
    while True:
        try:
            my_consumer = KafkaConsumer('a-metric', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - (data['timestamp'] + hour_8_sec) % 60
                if metric_d.get(t) is None:
                    metric_d[t] = []
                try:
                    metric_d[t].append(
                        [data['timestamp'], data['cmdb_id'], data['kpi_name'], data['value']])
                except Exception:
                    metric_d[t].append([data.get('timestamp'), data.get('cmdb_id'), data.get('kpi_name'),
                                        data.get('value')])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def trace():
    """
    consume a-trace
    """
    while True:
        try:
            my_consumer = KafkaConsumer('a-trace', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - (data['timestamp'] + hour_8_sec) % 60
                if trace_d.get(t) is None:
                    trace_d[t] = []
                try:
                    trace_d[t].append([data['timestamp'], data['cmdb_id'], data['parent_id'], data['span_id'],
                                       data['trace_id'], data['duration']])
                except Exception:
                    trace_d[t].append([data.get('timestamp'), data.get('cmdb_id'), data.get('parent_id'),
                                       data.get('span_id'), data.get('trace_id'), data.get('duration')])
                    print(traceback.format_exc())
        except Exception:
            print(traceback.format_exc())


def log():
    """
    consume a-log
    """
    while True:
        try:
            my_consumer = KafkaConsumer('a-log', bootstrap_servers=ips, auto_offset_reset='latest',
                                        enable_auto_commit=False,
                                        security_protocol='PLAINTEXT')
            for message in my_consumer:
                data = json.loads(message.value.decode('utf8'))
                data['timestamp'] = int(data['timestamp'])
                t = data['timestamp'] - (data['timestamp'] + hour_8_sec) % 60
                if log_d.get(t) is None:
                    log_d[t] = []
                try:
                    log_d[t].append([data['id'], data['timestamp'], data['cmdb_id'], data['logname'],
                                     data['value']])
                except Exception:
                    log_d[t].append([data.get('id'), data.get('timestamp'), data.get('cmdb_id'),
                                     data.get('logname'), data.get('value')])
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
        n = 0
        try:
            for d in [kpi_d, metric_d, trace_d, log_d]:
                print(n, len(d), next(iter(d.items()))[1][0])
                n += 1
        except Exception:
            pass
        time.sleep(t)


def data_deal():
    Thread(target=kpi).start()
    Thread(target=metric).start()
    Thread(target=trace).start()
    Thread(target=log).start()
    Thread(target=clean, args=[10]).start()
    # Thread(target=test, args=[10]).start()


if __name__ == '__main__':
    data_deal()
