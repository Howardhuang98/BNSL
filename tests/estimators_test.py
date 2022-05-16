#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/30 11:27  
------------      
"""
import unittest

import pandas as pd

import bnsl
from bnsl.estimators import L2C, HC, DP
from bnsl.graph import compare


class Test_estimator(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv(r"test_data/Asia.csv")[:2000]
        self.Asian = bnsl.DAG()
        self.Asian.read(r"test_data/test_save_edge_list.csv")

    def test_hc(self):
        hc = HC(self.data)
        hc.run()
        print(compare(hc.result, self.Asian))

    def test_dp(self):
        dp = DP(self.data)
        dp.run()
        print(compare(dp.result, self.Asian))

    def test_l2c(self):
        e0 = pd.read_csv(r"test_data/asia_expert0.csv", index_col=0)
        e1 = pd.read_csv(r"test_data/asia_expert1.csv", index_col=0)
        e2 = self.Asian.adj_df()

        e = [e2]
        est = L2C(self.data, e, [1])
        est.run(restart=5)
        print(compare(est.result, self.Asian))

        e = [e0, e2]
        est = L2C(self.data, e, [0.5, 0.5])
        est.run(restart=5)
        print(compare(est.result, self.Asian))

        e = [e0, e1, e2]
        est = L2C(self.data, e, [0.333, 0.333, 0.333])
        est.run(restart=5)
        print(compare(est.result, self.Asian))
