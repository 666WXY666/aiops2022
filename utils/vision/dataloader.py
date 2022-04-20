from datetime import datetime
import os
from posixpath import split
from matplotlib import container
import pandas as pd
import time

import scipy as sp

def make_data(path, type):

    """
    制作某一类指标的Dataframe(多csv合并、排序)

    Parameters
    --------
    path: string, 文件路径
        e.g. r'.\faults\tar\2022-03-24-cloudbed3\cloudbed-3\metric\service'
    type: string, 指标类型
        e.g. "container", "istio", "jvm", "node", "service"

    Return
    --------
    metric_df: pd.DataFrame, 指标Dataframe
    """

    if type=="service":
        file_name = os.listdir(path)[0]
        dir = os.path.abspath(path)
        file_path = os.path.join(dir, file_name)
        metric_service = pd.read_csv(file_path)
        metric_service.sort_values(by=['service','timestamp'],inplace=True)
        metric_df = metric_service.copy()
        metric_df['svc'] = metric_df['service'].apply(lambda x: x.split('-')[0])
        metric_df['method'] = metric_df['service'].apply(lambda x: x.split('-')[1])

    
    if type=="node":
        file_name = os.listdir(path)[0]
        dir = os.path.abspath(path)
        file_path = os.path.join(dir, file_name)
        metric_node = pd.read_csv(file_path)
        metric_node.sort_values(by=['cmdb_id','kpi_name', 'timestamp'],inplace=True)
        metric_df = metric_node.copy()
    
    if type=="container":
        file_list = os.listdir(path)
        csves = []
        for file_name in file_list:
            dir = os.path.abspath(path)
            file_path = os.path.join(dir, file_name)
            metric_container = pd.read_csv(file_path)
            csves.append(metric_container)
        metric_df = pd.concat(csves, axis=0)
        metric_df.sort_values(by=['cmdb_id','kpi_name', 'timestamp'],inplace=True)
        metric_df['node'] = metric_df['cmdb_id'].apply(lambda x: x.split('.')[0])
        metric_df['container'] = metric_df['cmdb_id'].apply(lambda x: x.split('.')[1])

    
    if type=="istio" or type=="jvm":
        file_list = os.listdir(path)
        csves = []
        for file_name in file_list:
            dir = os.path.abspath(path)
            file_path = os.path.join(dir, file_name)
            metric_cij = pd.read_csv(file_path)
            csves.append(metric_cij)
        metric_df = pd.concat(csves, axis=0)
        metric_df.sort_values(by=['cmdb_id','kpi_name', 'timestamp'],inplace=True)
    
    metric_df['datetime'] = metric_df['timestamp'].apply(lambda x: timestamp2datetime(x))
    return metric_df

def timestamp2datetime(timestamp):
    timestamp = time.localtime(timestamp) # struct_time
    format_time = time.strftime("%Y-%m-%d %H:%M:%S", timestamp)
    return format_time