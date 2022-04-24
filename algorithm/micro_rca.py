'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description:
Version:
Author: WangXingyu
Date: 2022-04-23 23:17:59
LastEditors: WangXingyu
LastEditTime: 2022-04-24 19:24:56
'''

import math
import os
import random
import re
import time
from inspect import trace
from pydoc import pager

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from torch import fix_
from tqdm import tqdm


# GMPG = np.load('pods_mpg.npy', allow_pickle=True)
class PageRCA():
    def __init__(self, ts, fDict, responds, metric):
        assert ts < 10**10
        self.BREAK_L = False
        self.ts = ts
        self.ts13 = ts*1000
        # self.traces = traces.sort_values(by=['timestamp'], ignore_index=True)
        self.responds = self.del_responds(responds)
        self.fDict = fDict
        self.metric = metric
        self.mpg_pd = pd.DataFrame(
            data=None, columns=['source', 'destination', 'operation_name'])
        self.pod_pd = self.get_pod_pd()
        self.fPods = self.del_fPods()

    def _get_fix_mpg_pd(self):
        responds = self.responds
        addE_pd = pd.DataFrame(data=None, columns=[
                               'source_destination_svc', 'operation_name', 'e_type', 'pod'])
        res_pd = pd.DataFrame(data=None, columns=[
                              'source', 'destination', 'operation_name'])
        for cmdb in responds['cmdb_id'].value_counts().index:
            cmdb_res = cmdb.split('.')
            pod = cmdb_res[0]
            et = cmdb_res[1]
            s = cmdb_res[2]
            d = cmdb_res[3]

            opt = 'ADD EDGE (from isito)'

            if s in ['unknown', 'jaeger-collector'] or d in ['unknown', 'jaeger-collector']:
                continue

            res = {
                'source_destination_svc': '{}_{}'.format(s, d),
                'operation_name': opt,
                'e_type': et,
                'pod': pod
            }
            addE_pd = pd.concat([addE_pd, pd.DataFrame(
                data=res, index=[0])], ignore_index=True)

        for s_d in addE_pd['source_destination_svc'].value_counts().index:
            res_pods = addE_pd.loc[addE_pd['source_destination_svc'] == s_d]
            if len(res_pods) >= 2:
                s_t = res_pods.loc[res_pods['e_type'] == 'source', 'pod']
                d_t = res_pods.loc[res_pods['e_type'] == 'destination', 'pod']
                if len(s_t) == 0 or len(d_t) == 0:
                    continue

                for i in range(min((len(s_t), len(d_t)))):
                    tmp_add = self._link_pd(s_t.iloc[i], d_t.iloc[i])
                    res_pd = pd.concat([res_pd, tmp_add], ignore_index=True)
                sub_sd = abs(len(s_t) - len(d_t))
                if sub_sd != 0:
                    if len(s_t) > len(d_t):
                        for i in range(sub_sd):
                            tmp_add = self._link_pd(
                                s_t.iloc[i+len(d_t)-1], d_t.iloc[0])
                            res_pd = pd.concat(
                                [res_pd, tmp_add], ignore_index=True)
                    else:
                        for i in range(sub_sd):
                            tmp_add = self._link_pd(
                                s_t.iloc[0], d_t.iloc[i+len(s_t)-1])
                            res_pd = pd.concat(
                                [res_pd, tmp_add], ignore_index=True)

        return res_pd

    def _link_pd(self, s, d, opt='ADD EDGE (from isito)'):
        res = {
            'source': s,
            'destination': d,
            'operation_name': opt
        }
        return pd.DataFrame(data=res, index=[0])

    def del_fPods(self):
        pos_pd = self.pod_pd
        fPods = self.fDict
        node, service, pod = fPods['node'], fPods['service'], fPods['pod']

        node_in_pods = []
        if len(node) != 0:
            for n in node:
                if n in pos_pd['node'].values:
                    node_in_pods.append(n)
            if len(node_in_pods) == 0:
                self.BREAK_L = True
                self.node = node
                return None

        nodes_pd = pd.DataFrame(data=None, columns=pos_pd.columns)
        svcs_pd = pd.DataFrame(data=None, columns=pos_pd.columns)
        pods_pd = pd.DataFrame(data=None, columns=pos_pd.columns)
        for n in node:
            node_pd = pos_pd.loc[pos_pd['node'] == n]
            nodes_pd = pd.concat([nodes_pd, node_pd], ignore_index=True)

        for s in service:
            if len(nodes_pd) == 0:
                tmp_pd = self.pod_pd
            else:
                tmp_pd = nodes_pd
            svc_pd = tmp_pd.loc[tmp_pd['svc'] == s]
            if(len(svc_pd) > 0):
                svcs_pd = pd.concat([svcs_pd, svc_pd], ignore_index=True)

        for p in pod:
            if len(svcs_pd) == 0 and len(nodes_pd) == 0:
                tmp_pd = self.pod_pd
            elif len(svcs_pd) == 0 and len(nodes_pd) != 0:
                tmp_pd = nodes_pd
            else:
                tmp_pd = svcs_pd
            pod_pd = tmp_pd.loc[tmp_pd['pod'] == p]
            if len(pod_pd) > 0:
                pods_pd = pd.concat([pods_pd, pod_pd], ignore_index=True)

        if len(pods_pd) == 0 and len(svcs_pd) == 0 and len(nodes_pd) == 0:
            return pod
        elif len(pods_pd) > 0:
            return pods_pd['pod'].tolist()
        elif len(svcs_pd) > 0:
            return svcs_pd['pod'].tolist()
        else:
            if len(pod) != 0:
                return pod
            elif len(service) != 0:
                return service
            else:
                return nodes_pd['pod'].tolist()

    def fix_mpg(self):
        fix_pd = self._get_fix_mpg_pd()
        mpg_pd = self.mpg_pd
        self.mpg_pd = pd.concat([mpg_pd, fix_pd], ignore_index=True)
        self.mpg_pd = fix_pd
        return self.mpg_pd

    def del_responds(self, responds):
        res_out = pd.DataFrame(data=None, columns={
                               'timestamp', 'cmdb_id', 'value'})
        responds_pd = responds.copy()
        ts = responds_pd['timestamp'].value_counts().index
        responds_pd['value'] = responds_pd['value'].astype('float')
        for t in ts:
            t_pd = responds_pd.loc[responds_pd['timestamp'] == t]
            cmdb_index = t_pd['cmdb_id'].value_counts().index
            for cmdb in cmdb_index:
                c_pd = t_pd.loc[t_pd['cmdb_id'] == cmdb]['value']
                if len(c_pd) > 0:
                    c_pd = c_pd.astype('float')
                ts = c_pd.sum() if len(c_pd) > 1 else c_pd.item()
                res = {
                    'timestamp': t,
                    'cmdb_id': cmdb,
                    'value': ts
                }
                res_out = pd.concat([res_out, pd.DataFrame(
                    data=res, index=[0])], ignore_index=True)
        res_out['value'] = res_out['value'].astype('float')
        res_out['timestamp'] = res_out['timestamp'].astype('int')

        return res_out

    def get_full_mpg(self):
        mpg_pd = self.fix_mpg()
        mpg = nx.DiGraph()
        print('Start to full construct mpg:')
        for i in tqdm(range(len(mpg_pd))):
            trace = mpg_pd.iloc[i]
            s, d = trace['source'], trace['destination']
            opt = trace['operation_name']
            if np.nan not in [s, d]:
                if s == d:
                    continue
                mpg.add_edge(s, d)
                mpg.nodes[s]['type'], mpg.nodes[d]['type'] = 'pod', 'pod'
                mpg.nodes[s]['IsFault'],  mpg.nodes[d]['IsFault'] = 0, 0
                mpg.edges[s, d]['opt'] = opt
                mpg.edges[s, d]['IsFault'] = 0
        self.mpg = mpg
        return mpg

    def get_mpg(self, map_pd):
        mpg_pd = map_pd
        mpg = nx.DiGraph()
        print('Start to construct mpg:')
        for i in tqdm(range(len(mpg_pd))):
            trace = mpg_pd.iloc[i]
            s, d = trace['source'], trace['destination']
            opt = trace['operation_name']
            if np.nan not in [s, d]:
                if s == d:
                    continue
                mpg.add_edge(s, d)
                mpg.nodes[s]['type'], mpg.nodes[d]['type'] = 'pod', 'pod'
                mpg.nodes[s]['IsFault'],  mpg.nodes[d]['IsFault'] = 0, 0
                mpg.edges[s, d]['opt'] = opt
                mpg.edges[s, d]['IsFault'] = 0
        return mpg

    def get_sub_mpg(self):
        mpg_pd = self.mpg_pd
        ePodlist = self.fPods
        sub_mpg_pd = pd.DataFrame(
            data=None, columns=['source', 'destination', 'operation_name'])
        for pod in ePodlist:
            res = mpg_pd.loc[(mpg_pd['source'] == pod) |
                             (mpg_pd['destination'] == pod)]
            sub_mpg_pd = pd.concat([sub_mpg_pd, res], ignore_index=True)

        sub_mpg = self.get_mpg(sub_mpg_pd)
        for n, _ in sub_mpg.nodes.items():
            sub_mpg.nodes[n]['IsFault'] = 1 if n in ePodlist else 0
        for e, _ in sub_mpg.edges.items():
            if (e[0] in ePodlist) or (e[1] in ePodlist):
                sub_mpg.edges[e]['IsFault'] = 1
            else:
                sub_mpg.edges[e]['IsFault'] = 0
        self.sub_mpg = sub_mpg
        return sub_mpg

    def get_edge_w(self, alpha=0.55):
        sub_mpg = self.sub_mpg
        res_pd = self.responds
        # timestamp = self.ts
        # startT = timestamp-5*60*1000
        # res_pd = res_pd.loc[(res_pd['timestamp']>=startT) &(res_pd['timestamp']<=timestamp)]
        corr_sum, num = 0, 0
        zeroL = []
        for edge, opt in sub_mpg.edges.items():
            if (sub_mpg.nodes[edge[0]]['IsFault'] == 1) and (sub_mpg.nodes[edge[1]]['IsFault'] == 1):
                sub_mpg.edges[edge]['weight'] = alpha
                continue

            s, d = edge[0], edge[1]
            s_v, d_v = s.split('-')[0], d.split('-')[0]
            s_cmdb_id = "{}.source.{}.{}".format(s, s_v, d_v)
            d_cmdb_id = "{}.destination.{}.{}".format(d, s_v, d_v)
            s_t = res_pd.loc[res_pd['cmdb_id'] == s_cmdb_id]['value']
            d_t = res_pd.loc[res_pd['cmdb_id'] == d_cmdb_id]['value']

            if (len(s_t) == 0) or (len(d_t) == 0):
                if len(s_t) == len(d_t):
                    sub_mpg.edges[edge]['weight'] = 0.01
                else:
                    zeroL.append(edge)
                continue

            s_t, d_t = s_t.reset_index(drop=True), d_t.reset_index(drop=True)

            if len(s_t) > len(d_t):
                s_t = s_t[0:len(d_t)]
            elif len(s_t) < len(d_t):
                d_t = d_t[0:len(s_t)]
            corr_t = s_t.corr(d_t)

            if math.isnan(corr_t):
                if s_t.mean() == d_t.mean():
                    sub_mpg.edges[edge]['weight'] = 0.9
                else:
                    zeroL.append(edge)
                continue

            sub_mpg.edges[edge]['weight'] = abs(corr_t)
            corr_sum += abs(corr_t)
            num += 1

        if num != 0:
            corr_sum /= num
            for e in zeroL:
                sub_mpg.edges[e]['weight'] = corr_sum

        self.sub_mpg = sub_mpg

        return sub_mpg

    def _remove_zero(self, se):
        res = []
        for val in se.values:
            if val != 0:
                res.append(val)
        return pd.Series(res)

    def get_p_nal(self):
        sub_mpg = self.sub_mpg
        PodsList = []
        personalization = {}
        for n, d in sub_mpg.nodes.items():
            PodsList.append(n)
        for e_n in PodsList:
            num = 0
            e_w = 0.0
            for e, d in sub_mpg.edges.items():
                if e_n in e:
                    e_w += d['weight']
                    num += 1
            if num != 0:
                e_w /= num
            personalization[e_n] = e_w
        self.personalization = personalization
        return personalization

    def get_pod_metrc(self, pod):
        podL = []
        for n, d in self.sub_mpg.nodes.items():
            podL.append(n)
        metric = self.metric
        metric = metric.apply(lambda x: pd.Series(
            x_v.split('.')[-1] for x_v in x) if x.name == 'cmdb_id' else x)
        cpu = 'container_cpu_usage_seconds'
        mem = 'container_memory_usage_MB'
        net = 'container_network_transmit_packets.eth0'
        podCPU = metric.loc[(metric['cmdb_id'] == pod) &
                            (metric['kpi_name'] == cpu)]
        podMEM = metric.loc[(metric['cmdb_id'] == pod) &
                            (metric['kpi_name'] == mem)]
        podNET = metric.loc[(metric['cmdb_id'] == pod) &
                            (metric['kpi_name'] == net)]
        podCPU = podCPU.reset_index(drop=True)
        podMEM = podMEM.reset_index(drop=True)
        podNET = podNET.reset_index(drop=True)

        return podCPU, podMEM, podNET

    def get_pod_la(self):
        sub_mpg = self.sub_mpg
        responds = self.responds
        res_dir = {}
        for n, d in sub_mpg.nodes.items():
            res_dir[n] = []

        for edge, d in sub_mpg.edges.items():
            s, d = edge[0], edge[1]
            s_v, d_v = s.split('-')[0], d.split('-')[0]
            s_cmdb_id = "{}.source.{}.{}".format(s, s_v, d_v)
            d_cmdb_id = "{}.destination.{}.{}".format(d, s_v, d_v)
            s_t = responds.loc[responds['cmdb_id']
                               == s_cmdb_id]['value'].to_list()
            d_t = responds.loc[responds['cmdb_id']
                               == d_cmdb_id]['value'].to_list()
            res_dir[s].append(s_t)
            res_dir[d].append(d_t)

        for pod, data in res_dir.items():
            tmp_num = np.array(res_dir[pod])
            res_dir[pod] = pd.Series(tmp_num.mean(axis=0))

        self.res_dir = res_dir
        return res_dir

    def get_pern_f(self):
        podLlist = self.get_pod_la()
        sub_mpg = self.sub_mpg
        personalization = self.personalization
        p = personalization
        for pod, d in sub_mpg.nodes.items():
            podCPU, podMEM, podNET = self.get_pod_metrc(pod)
            corr_cpu = abs(podCPU['value'].corr(podLlist[pod]))
            corr_mem = abs(podMEM['value'].corr(podLlist[pod]))
            corr_net = abs(podNET['value'].corr(podLlist[pod]))
            corrL = []
            for i in [corr_cpu, corr_mem, corr_net]:
                if math.isnan(i):
                    corrL.append(0.01)
                else:
                    corrL.append(i)
            personalization[pod] *= max(corrL)
            personalization[pod] /= self.mpg.degree(pod)
            # if 'checkoutservice' in pod or 'paymentservice' in pod:
            #     personalization[pod] *= 0.1
            if pod in self.fPods:
                personalization[pod] *= 2.0
        self.personalization = personalization

        return personalization

    def get_pods_s(self):
        sub_mpg = self.sub_mpg
        personalization = self.get_p_nal()
        personalization = self.get_pern_f()
        sub_mpg = sub_mpg.reverse(copy=True)
        anomaly_score = nx.pagerank(
            sub_mpg, alpha=0.85, personalization=personalization, max_iter=10000)
        anomaly_score = sorted(anomaly_score.items(),
                               key=lambda x: x[1], reverse=True)
        return anomaly_score

    def get_pod_pd(self):
        metric = self.metric
        cmdbL = metric['cmdb_id'].value_counts().index
        res_pd = pd.DataFrame(data=None, columns=['node', 'svc', 'pod'])
        for cmdb in cmdbL:
            node = cmdb.split('.')[0]
            pod = cmdb.split('.')[-1]
            svc = pod.split('-')[0]
            res = {
                'node': node,
                'svc': svc,
                'pod': pod
            }
            res_pd = pd.concat([res_pd, pd.DataFrame(
                res, index=[0])], ignore_index=True)
        self.pod_pd = res_pd
        return res_pd

    def res_out(self, etype='node'):
        anomaly_score = self.anomaly_score
        pos_pd = self.pod_pd
        res_out = {}
        for pods in anomaly_score:
            pod, score = pods[0], pods[1]
            res = pos_pd.loc[pos_pd['pod'] == pod, etype].item()
            if res not in res_out.keys():
                res_out[res] = score
            else:
                res_out[res] += score

        return res_out

    #     res_out = dict(
    #         sorted(res_out.items(), key=lambda x: x[1], reverse=True))

    #     if etype != 'node':
    #         FRONTEND = False
    #         CHECKOUT = False
    #         PAYMENT = False
    #         for p in self.fPods:
    #             if 'frontend' in p:
    #                 FRONTEND = True
    #             if 'checkout' in p:
    #                 CHECKOUT = True
    #             if 'payment' in p:
    #                 PAYMENT = True

    #         out0 = list(res_out.keys())
    #         if FRONTEND:
    #             out1 = [i for i in out0 if 'frontend' not in i]
    #             if len(out1) == 0:
    #                 return out0[0]
    #             else:
    #                 if CHECKOUT:
    #                     out2 = [i for i in out1 if 'checkout' not in i]
    #                     if len(out2) == 0:
    #                         return out1[0]
    #                     else:
    #                         if PAYMENT:
    #                             out3 = [i for i in out2 if 'payment' not in i]
    #                             if len(out3) == 0:
    #                                 return out2[0]
    #                             else:
    #                                 return out3[0]
    #                         else:
    #                             return out2[0]
    #                 else:
    #                     return out1[0]
    #         else:
    #             return out0[0]

    def rule_res_out(self, res_out, etype):
        res_out = dict(
            sorted(res_out.items(), key=lambda x: x[1], reverse=True))
        if etype != 'node':
            podsLevel = {
                'frontend': 1,
                'checkout': 2,
                'payment': 3
            }
            max_s = 0
            for p in self.fPods:
                pSvc = p.split('-')[0]
                key_res = None
                for key in podsLevel.keys():
                    res = re.match(key, pSvc)
                    if res != None:
                        key_res = key
                if key_res == None:
                    max_s = 4
                else:
                    sc = podsLevel[key_res]
                    max_s = max_s if max_s > sc else sc

            for r, s in res_out.items():
                pSvc = r.split('-')[0]
                key_res = None
                for key in podsLevel.keys():
                    res = re.match(key, pSvc)
                    if res != None:
                        key_res = key
                if key_res == None:
                    return r
                else:
                    sc = podsLevel[key_res]
                    if sc == max_s:
                        return r
            return list(res_out.keys())[0]
        else:
            return list(podsLevel.keys())[0]

    def roulette_out(self, res):
        nameL = list(res.keys())
        scores = list(res.values())
        scores_s = []
        sum_s = sum(scores)
        sumS = 0
        for idx, s in enumerate(scores):
            scores_s.append(sumS)
            sumS += scores[idx]

        s = np.random.uniform(low=0, high=1)
        res_out = nameL[0]
        for i in range(len(scores_s) - 1):
            if scores_s[i] <= s and scores_s[i+1] >= s:
                res_out = nameL[i]
        if s > scores_s[-1]:
            res_out = nameL[-1]

        return res_out

    def svc_add_rule(self, etype):
        if etype == 'svc':
            return 'svc'
        fPods = self.fPods
        pod_pd = self.pod_pd
        svcList = pod_pd['svc'].value_counts().index.tolist()
        svc_pods_num = pod_pd['svc'].value_counts().values.tolist()
        for svc, num in zip(svcList, svc_pods_num):
            p_num = 0
            for p in fPods:
                p_svc = p.split('-')[0]
                if p_svc == svc:
                    p_num += 1
                    if p_num == num:
                        return 'svc'

        return etype

    def do_rca(self, pos_pd=None):
        fDict = self.fDict
        if len(fDict['node']) != 0:
            ftype = 'node'
        elif len(fDict['service']) != 0:
            ftype = 'svc'
        else:
            ftype = 'pod'
        if not self.BREAK_L:
            ftype = self.svc_add_rule(etype=ftype)

            print('Start RCA:\nTime:{}\nfPods:{}'.format(self.ts, self.fPods))
            mpg = self.get_full_mpg()
            sub_mpg = self.get_sub_mpg()
            sub_mpg = self.get_edge_w()
            self.anomaly_score = self.get_pods_s()
            print("Anomaly_score:{}".format(self.anomaly_score))

            out = self.res_out(etype=ftype)
            ans_out = self.rule_res_out(out, etype=ftype)

            print("fPods: {}".format(self.fPods))
            if ftype == 'svc':
                ans_out = [i for i in ans_out if i >= 'A' and i <= 'z']

            return ans_out
        else:
            return self.node


class ScanMpg():
    def __init__(self, ts, traces):
        assert ts >= 10*10
        assert 'timestamp' in traces.columns
        self.ts = ts
        self.ts13 = ts*1000
        self.traces = traces.sort_values(by=['timestamp'], ignore_index=True)
        self.mpg_pd = pd.DataFrame(
            data=None, columns=['source', 'destination', 'operation_name'])

    def _findIndex(self, index, exp, endNum):
        if index < exp//2:
            return (0, exp)
        elif index < endNum-exp:
            return (index-exp//2, index+exp//2)
        else:
            return (endNum-exp, endNum)

    def get_traces_map(self, expTime=3):
        traces = self.traces
        mpg_pd = self.mpg_pd
        endTs = self.ts13
        startTs = endTs - expTime*60*1000
        traces_s = traces.loc[(traces['timestamp'] >= startTs) & (
            traces['timestamp'] <= endTs)]
        traces_s = traces_s.sort_values(by=['trace_id'], ignore_index=True)

        for i in tqdm(range(len(traces_s))):
            trace = traces_s.iloc[i]
            traceID = trace['trace_id']
            traces_tmp = traces_s.loc[self._findIndex(
                i, 72, len(traces_s)-1), :]
            source = trace['cmdb_id']
            opt = trace['operation_name']
            pSpanID = trace['parent_span']

            if pSpanID == np.nan:
                destination = 'root'
            else:
                pSpan = traces_tmp.loc[traces_tmp['span_id']
                                       == pSpanID, 'cmdb_id']
                destination = pSpan.item() if len(pSpan) != 0 else 'Not found'

            if destination == 'Not found':
                continue
            res = {
                'source': source,
                'destination': destination,
                'operation_name': opt
            }
            f_t = mpg_pd.loc[(mpg_pd['source'] == source) & (
                mpg_pd['destination'] == destination)]
            if len(f_t) != 0:
                continue
            else:
                mpg_pd = pd.concat([mpg_pd, pd.DataFrame(
                    data=res, index=[0])], ignore_index=True)

        GMPG = mpg_pd.to_numpy()
        np.save('pods_mpg.npy', GMPG)
        return mpg_pd

    def do_scan(self, expTime=10):
        print("Update MPG of pods!")
        res = self.get_traces_map(expTime=10)
        print('In {}. NUM of MPG\'s edges: {}'.format(self.ts, len(res)))


if __name__ == '__main__':
    MODE_SCANS = False
    testF = {'timestamp': 1647809681,
             'type': 'pod',
             'cmdb_id': 'recommendationservice-2',
             'fault': 'k8s容器网络丢包'}

    trace_pd = pd.read_csv('tracs_sort_times.csv', index_col=0)
    res_pd = pd.read_csv(
        './cloudbed-1/metric/istio/kpi_istio_request_duration_milliseconds.csv')
    endT = testF['timestamp']
    startT = endT - 10*60

    traces = trace_pd.loc[(trace_pd['timestamp'] >= startT*1000)
                          & (trace_pd['timestamp'] <= endT*1000)]
    traces = traces.reset_index(drop=True)
    responds = res_pd.loc[(res_pd['timestamp'] >= startT)
                          & (res_pd['timestamp'] <= endT)]
    responds = responds.reset_index(drop=True)

    if MODE_SCANS:
        scanM = ScanMpg(endT, traces)
        scanM.do_scan()

    else:
        pageRca = PageRCA(ts=endT, traces=traces, responds=responds, fPods=[
                          'recommendationservice-2', 'recommendationservice-1'])
        res = pageRca.do_rca()
        pass
