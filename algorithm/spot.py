'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-23 18:49:49
LastEditors: WangXingyu
LastEditTime: 2022-04-27 19:34:46
'''
from math import floor, log

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

            Parameters
            ----------
            q
                    Detection level (risk)

            Returns
            ----------
        SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n-self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (
                    r, 100*r/self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
            ----------
            init_data : list, numpy.array or pandas.Series
                    initial batch to calibrate the algorithm

        data : numpy.array
                    data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data*data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
            ----------
            data : list, numpy.array, pandas.Series
                    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, verbose=False):
        """
        Run the calibration (initialization) step

        Parameters
            ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t 
            verbose : bool
                    (default = True) If True, gives details about the batch initialization
        """
        level = level-floor(level)

        n_init = self.init_data.size

        # we sort X to get the empirical quantile
        S = np.sort(self.init_data)
        # t is fixed for the whole algorithm
        self.init_threshold = S[int(level*n_init)]

        # initial peaks
        self.peaks = self.init_data[self.init_data >
                                    self.init_threshold]-self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t'+chr(0x03B3) + ' = ' + str(g))
            print('\t'+chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' %
                  (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
                    scalar function 
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1]-bounds[0])/(npoints+1)
            X0 = np.arange(bounds[0]+step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g+fx**2
                j[i] = 2*fx*jac(x)
                i = i+1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds]*len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
                    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma/sigma
            L = -n * log(sigma) - (1 + (1/gamma)) * (np.log(1+tau*Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
                    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1/s)

        def w(Y, t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            return us*vs-1

        def jac_w(Y, t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            jac_us = (1/t)*(1-vs)
            jac_vs = (1/t)*(-vs+np.mean(1/s**2))
            return us*jac_vs+vs*jac_us

        # In case of "zero-size array"
        if self.Nt == 0:
            Ym, YM, Ymean = 0, 0, 0
        else:
            Ym = self.peaks.min()
            YM = self.peaks.max()
            Ymean = self.peaks.mean()

        a = -1/YM
        if abs(a) < 2*epsilon:
            epsilon = abs(a)/n_points

        a = a + epsilon
        b = 2*(Ymean-Ym)/(Ymean*Ym)
        c = 2*(Ymean-Ym)/(Ym**2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a+epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1+z*self.peaks)-1
            sigma = gamma/z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
                    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma/gamma)*(pow(r, -gamma)-1)
        else:
            return self.init_threshold - sigma*log(r)

    def run(self, with_alarm=True):
        """
        Run SPOT on the stream

        Parameters
        ----------
        with_alarm : bool
                    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'

            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms

        """
        if (self.n > self.init_data.size):
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        fault_begin = -1
        # Loop over the stream
        # for i in tqdm.tqdm(range(self.data.size), ascii=True):
        for i in range(self.data.size):

            # If the observed value exceeds the current threshold (alarm case)
            if self.data[i] > self.extreme_quantile:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks = np.append(
                        self.peaks, self.data[i]-self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] > self.init_threshold:
                # we add it in the peaks
                self.peaks = np.append(
                    self.peaks, self.data[i]-self.init_threshold)
                self.Nt += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
            else:
                self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        for i in range(len(alarm)):
            if i+3 <= len(alarm)-1:
                if alarm[i+2]-alarm[i] == 2:
                    fault_begin = alarm[i]
                    break

        return {'thresholds': th, 'alarms': alarm, 'fault_begin': fault_begin}

    def train(self, norm_data):
        """
        Generate & Save threshold for cmdbs

        Parameters:
        ----------
        norm_data: pd.DataFrame
            Normal data of each cmdb, shape: 1440x57
        """
        cmdb = norm_data.columns.tolist()  # 指标列表
        threshold_list = {}
        for id in cmdb:
            # try:
            data = norm_data[id].values
            # if id == 'cartservice-grpc':
            #     n_init = 1000
            # else:
            n_init = 1440
            # n_init = int(len(data)*3/4)
            init_data = data[1:n_init]
            _data = data[n_init:]
            self.fit(init_data, _data)
            self.initialize()
            results = self.run()
            res_thre = results['thresholds']
            threshold_list[id] = res_thre[-1]*1.5
            # except:
            #     print(id)

        thre_df = pd.DataFrame(threshold_list, index=[0])
        joblib.dump(thre_df, './model/spot/spot.pkl')

    def detect(self, stream_data):
        """
        Stream data anomaly detection

        Parameters:
        ----------
        stream_data: pd.DataFrame
            Aggregated data from kafka, shape: 1x57 
        """

        # Save threshold dict
        threshold_df = joblib.load('./model/spot/spot.pkl')
        threshold_list = []
        thre_cols = threshold_df.columns
        for col in thre_cols:
            threshold_list.append(threshold_df[col].values[0])
        threshold_list = np.array(threshold_list)

        # Save stream data
        stream_list = []
        stream_cols = stream_data.columns
        for col in stream_cols:
            stream_list.append(stream_data[col].values[0])
        stream_list = np.array(stream_list)

        # Compare
        result = stream_list > threshold_list  # True-False列表, cmdb顺序参照SPOT类的cmdb_list属性
        return result

    def check_anomaly(self, abn_dict):
        anomaly_dict = {'service': [], 'pod': [], 'node': []}
        fault_flag = False  # 判定是否异常条件: 连续两个点异常
        for key, value in abn_dict.items():
            if value >= 2:
                words = key.split('-')
                if words[0] == 'node':
                    anomaly_dict['node'].append(key)
                elif (words[1] == 'grpc' or words[1] == 'http'):
                    if words[0] not in anomaly_dict['service']:
                        anomaly_dict['service'].append(words[0])  # 服务合并
                else:
                    anomaly_dict['pod'].append(key)
        if (len(anomaly_dict['service']) != 0 or len(anomaly_dict['pod']) != 0 or len(anomaly_dict['node']) != 0):
            fault_flag = True
        return fault_flag, anomaly_dict
