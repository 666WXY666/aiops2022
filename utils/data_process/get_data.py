'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-23 18:28:45
LastEditors: WangXingyu
LastEditTime: 2022-04-23 18:41:53
'''
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import schedule

from consumer import data_deal, kpi_d, metric_d


def main():
    current_time = int(time.time())
    print(time.strftime('%H:%M:%S', time.localtime(current_time)))
    current_time = current_time - current_time % 60
    print(current_time)

    kpi_list = kpi_d.get(current_time - 60, [])
    df_kpi = pd.DataFrame(
        kpi_list, columns=['timestamp', 'cmdb_id', 'kpi_name', 'value'])
    df1 = pd.read_csv('kpi.csv')
    df1 = pd.concat([df1, df_kpi]).reset_index(drop=True)
    df1.to_csv('kpi.csv', index=False)

    metric_list = metric_d.get(current_time - 60, [])
    df_service = pd.DataFrame(metric_list, columns=[
        'service', 'timestamp', 'rr', 'sr', 'count', 'mrt'])
    df2 = pd.read_csv('service.csv')
    df2 = pd.concat([df2, df_service]).reset_index(drop=True)
    df2.to_csv('service.csv', index=False)


if __name__ == '__main__':
    data_deal()
    schedule.every().minute.at(':59').do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)
