'''
Copyright: Copyright (c) 2021 WangXingyu All Rights Reserved.
Description: 
Version: 
Author: WangXingyu
Date: 2022-04-23 21:47:33
LastEditors: WangXingyu
LastEditTime: 2022-04-23 21:48:51
'''
import os

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


class CatBoost:
    def __init__(self, n_splits=5, random_seed=42,
                 model_path='./model/catboost/model/', log_path='./model/catboost/log/'):
        self.n_splits = n_splits
        self.model_path = model_path
        self.log_path = log_path
        self.random_seed = random_seed
        self.kf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self, X, y):
        for k, (train_index, valid_index) in enumerate(self.kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            model = CatBoostClassifier(loss_function='MultiClass', verbose=False,
                                       random_seed=self.random_seed, use_best_model=True,
                                       learning_rate=0.1, train_dir=self.log_path)
            model.fit(X_train, y_train, eval_set=(
                X_valid, y_valid), plot=False)
            joblib.dump(model, self.model_path + 'model_' + str(k) + '.pkl')

            out = model.predict(X)
            print("k: {0} || F1: {1:.4f}".format(
                k, f1_score(y, out, average='micro')))

    def test(self, X):
        outs = []
        for k in range(self.n_splits):
            model = joblib.load(self.model_path + 'model_' + str(k) + '.pkl')
            out = model.predict(X)
            outs.append(out)

        outs = np.array(outs).reshape(self.n_splits, -1).T
        outs_final = []
        for out in outs:
            outs_final.append(np.argmax(np.bincount(out)))
        return outs_final
