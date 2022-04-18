import os
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
from plotly import tools
import time

# trans
def timestamp2datetime(timestamp):
    timestamp = time.localtime(timestamp) # struct_time
    format_time = time.strftime("%Y-%m-%d %H:%M:%S", timestamp)
    return format_time

# Get infomation about a fault according to label.csv & timestamp 
def format_fault(faults_df, timestamp):
    """
    构造故障信息字典

    Parameters:
    --------
    file_path: pd.Df, Label Dataframe
    timestamp: int, 故障时间戳

    Return:
    --------
    fault_dict: dict, 故障信息字典
    """
    fault_dict = {}
    info = faults_df[faults_df['timestamp'] == timestamp]
    fault_dict['timestamp'] = info['timestamp'].values[0]
    fault_dict['level'] = info['level'].values[0]
    fault_dict['cmdb_id'] = info['cmdb_id'].values[0]
    fault_dict['failure_type'] = info['failure_type'].values[0]
    fault_dict['datetime'] = info['datetime'].values[0]
    return fault_dict

# Make plotly title
def get_fault_info(fault_dict):
    """
    生成图标题
    """
    return "Datetime: " + fault_dict['datetime'] + "    " + "Level/cmdb_id: " + fault_dict['level'] + "/" + fault_dict['cmdb_id'] + "    " + "Type: " + fault_dict['failure_type'] 

# Plot failure
def plot_rela(fault_dict, win_len, container_df, service_df, node_df):
    """
    根据某条故障绘制与该故障相关的指标图

    Parameters:
    --------
    fault_dict: dict, 某条故障的信息字典
    win_len: int, 时间窗
    xxx_df: make_data()得到的指标Dataframe
    
    Example:
    --------
    # timestamp = 1648104900
    # fault_test = format_fault(label, timestamp)
    # plot_rela(fault_test, 3600, metric_container, metric_service, metric_node)

    """

    fault_time = fault_dict['timestamp']
    half_win = win_len/2 # 绘制故障时间前后 half_win 的指标数据
    metric_container = container_df.copy()
    metric_service = service_df.copy()
    metric_node = node_df.copy()

    # 故障标记
    ftime = timestamp2datetime(fault_dict['timestamp'])
    start_time = timestamp2datetime(fault_dict['timestamp']-600) # 标记故障时间前后10min的窗口
    end_time = timestamp2datetime(fault_dict['timestamp']+600)
    title = get_fault_info(fault_dict)

    # 保存文件名
    fname = fault_dict['cmdb_id'] + "_" + fault_dict['failure_type']
    # 检测是否存在, 存在则加上后缀
    isExist = os.path.exists(fname+".html")
    while isExist:
        fname = fname + "_1"
        isExist = os.path.exists(fname+".html")


    if fault_dict['level']=='pod':
        
        container_name = fault_dict['cmdb_id'] # service-x
        svc_name = container_name.split('-')[0] # service
        node_name = metric_container[metric_container['container']==container_name].node.unique()[0] # node-x

        # Service指标
        traces_svc = []
        svc_kpi = ['rr', 'sr', 'mrt', 'count']
        df_svc = metric_service[metric_service['svc']==svc_name].copy()
        df_svc = df_svc[(df_svc['timestamp']>=fault_time-half_win)&(df_svc['timestamp']<=fault_time+half_win)]
        for kpi in svc_kpi:
            trace = go.Scatter(
                    x = df_svc['datetime'],y=df_svc[kpi],
                    mode = "lines",
                    name = kpi,
                    line = {'color': 'sandybrown', 'dash': 'solid'},
                    showlegend = True,
                )
            traces_svc.append(trace)

        # Container 指标
        traces_container = []
        df_container = metric_container[metric_container['container']==container_name].copy()
        container_kpi = df_container.kpi_name.unique().tolist()
        # print(len(container_kpi)) # 64
        for kpi in container_kpi:
            df = df_container[df_container['kpi_name']==kpi]
            # print(df.shape) # 1440,7
            df = df[(df['timestamp']>=fault_time-half_win)&(df['timestamp']<=fault_time+half_win)]
            trace = go.Scatter(
                    x = df['datetime'],y=df['value'],
                    mode = "lines",
                    name = kpi,
                    line = {'color': 'cornflowerblue', 'dash': 'solid'},
                )
            traces_container.append(trace)
        # print(len(traces_container)) # 64

        # Node 指标
        traces_node = []
        df_node = metric_node[metric_node['cmdb_id']==node_name].copy()
        node_kpi = df_node.kpi_name.unique().tolist()
        # print(len(node_kpi)) # 57
        for kpi in node_kpi:
            df = df_node[df_node['kpi_name']==kpi]
            # print(df.shape) # 1440,7
            df = df[(df['timestamp']>=fault_time-half_win)&(df['timestamp']<=fault_time+half_win)]
            trace = go.Scatter(
                    x = df['datetime'],y=df['value'],
                    mode = "lines",
                    name = kpi,
                    line = {'color': 'mediumseagreen', 'dash': 'solid'},
                )
            traces_node.append(trace)
        # print(len(traces_node)) # 57
        node_kpi = [node_name + " " +i for i in node_kpi ]

        titles = svc_kpi + container_kpi + node_kpi

        print("Start ploting..... File will be saved as: " + fname + ".html")

        fig_service = py.subplots.make_subplots(
            rows=32,cols=4,
            subplot_titles=tuple(titles)
            )
        r, c = 1, 1
        for i in range(len(traces_svc)):
            fig_service.append_trace(traces_svc[i],r,c) # 第r行第c列
            c = c + 1
        r, c = 2, 1
        for i in range(len(traces_container)):
            fig_service.append_trace(traces_container[i],r,c) # 第r行第c列
            if c<4:
                c = c+1
            else:
                r = r + 1
                c = 1
        r, c = 18,1
        for i in range(len(traces_node)):
            fig_service.append_trace(traces_node[i],r,c) # 第r行第c列
            if c<4:
                c = c+1
            else:
                r = r + 1
                c = 1

        fig_service.update_layout(
            title=title,
            title_font_size=24,
            height=6400,
        )
    
    if fault_dict['level']=='service':

        svc_name = fault_dict['cmdb_id'] # service
        container_name = [svc_name+i for i in ['-0', '-1', '-2', '2-0']] # service-x
        # node_name = metric_container[metric_container['container']==container_name].node.unique()[0] # node-x
        
        # Service指标
        traces_svc = []
        svc_kpi = ['rr', 'sr', 'mrt', 'count']
        df_svc = metric_service[metric_service['svc']==svc_name].copy()
        df_svc = df_svc[(df_svc['timestamp']>=fault_time-half_win)&(df_svc['timestamp']<=fault_time+half_win)]
        for kpi in svc_kpi:
            trace = go.Scatter(
                    x = df_svc['datetime'],y=df_svc[kpi],
                    mode = "lines",
                    name = kpi,
                    line = {'color': 'sandybrown', 'dash': 'solid'},
                    showlegend = True,
                )
            traces_svc.append(trace)
        
        # Container 指标
        traces_container = []
        container_kpi_total = []
        line_colors = ['cornflowerblue', 'darkcyan', 'limegreen', 'plum']
        color_id = 0
        for container in container_name:
            df_container = metric_container[metric_container['container']==container].copy()
            # 指标名称
            container_kpi = df_container.kpi_name.unique().tolist()
            for k in container_kpi:
                container_kpi_total.append(container + " " + k)
            # print(len(container_kpi)) # 64

            for kpi in container_kpi:
                df = df_container[df_container['kpi_name']==kpi]
                # print(df.shape) # 1440,7
                df = df[(df['timestamp']>=fault_time-half_win)&(df['timestamp']<=fault_time+half_win)]
                trace = go.Scatter(
                        x = df['datetime'],y=df['value'],
                        mode = "lines",
                        name = container + " " + kpi,
                        line = {'color': line_colors[color_id], 'dash': 'solid'},
                    )
                traces_container.append(trace)
            
            color_id = color_id+1
            # print(len(traces_container)) # 64

        titles = svc_kpi + container_kpi_total # 4 + 64*4

        print("Start ploting..... File will be saved as: " + fname + ".html")

        fig_service = py.subplots.make_subplots(
            rows=65,cols=4,
            subplot_titles=tuple(titles)
            )
        r, c = 1, 1
        for i in range(len(traces_svc)):
            fig_service.append_trace(traces_svc[i],r,c) # 第r行第c列
            c = c + 1
        r, c = 2, 1
        for i in range(len(traces_container)):
            fig_service.append_trace(traces_container[i],r,c) # 第r行第c列
            if c<4:
                c = c+1
            else:
                r = r + 1
                c = 1

        fig_service.update_layout(
            title=title,
            title_font_size=24,
            height=12800,width=2600
        )

    if fault_dict['level']=='node':
 
        node_name = fault_dict['cmdb_id']

        # Node 指标
        traces_node = []
        df_node = metric_node[metric_node['cmdb_id']==node_name].copy()
        node_kpi = df_node.kpi_name.unique().tolist()
        # print(len(node_kpi)) # 57
        for kpi in node_kpi:
            df = df_node[df_node['kpi_name']==kpi]
            # print(df.shape) # 1440,7
            df = df[(df['timestamp']>=fault_time-half_win)&(df['timestamp']<=fault_time+half_win)]
            trace = go.Scatter(
                    x = df['datetime'],y=df['value'],
                    mode = "lines",
                    name = kpi,
                    line = {'color': 'mediumseagreen', 'dash': 'solid'},
                )
            traces_node.append(trace)
        # print(len(traces_node)) # 57

        titles = node_kpi

        print("Start ploting..... File will be saved as: " + fname + ".html")

        fig_service = py.subplots.make_subplots(
            rows=16,cols=4,
            subplot_titles=tuple(titles)
            )

        r, c = 1, 1
        for i in range(len(traces_node)):
            fig_service.append_trace(traces_node[i],r,c) # 第r行第c列
            if c<4:
                c = c+1
            else:
                r = r + 1
                c = 1

        fig_service.update_layout(
            title=title,
            title_font_size=24,
            height=3200,
        )
    
    fig_service.add_vrect(x0=start_time, x1=end_time, fillcolor="red", opacity=0.1)
    fig_service.add_vline(x=ftime, line_width=2, line_dash="dash", line_color="crimson")
    py.offline.plot(fig_service,filename=fname+".html")
    print("Finish!")