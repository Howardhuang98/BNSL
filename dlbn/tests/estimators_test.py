#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/30 11:27  
------------      
"""
import os
import unittest

import pandas as pd

from dlbn.estimators import DP, HC, GA, KBNL, K2
from dlbn.graph import DAG
from dlbn.score import MDL_score, BIC_score


class Test_estimator(unittest.TestCase):
    """
    sample0.xlsx has the same value for E, F variable, thus there should be an edge between E and F, but
    unable to determine the direction.
    """

    def test_dp(self):
        data = pd.read_excel(r"../../datasets/tests/sample0.xlsx")
        dag = DAG()
        score = dag.score(MDL_score, data)
        dp = DP(data)
        dp.run()
        print(dp.result.edges)
        self.assertTrue(dp.result.score(MDL_score, data) <= score)

    def test_hc(self):
        # test small data
        data = pd.read_csv(r"./test_data/Asia.csv")
        hc = HC(data)
        hc.run(restart=3)
        print(hc.result.calculated_score)

        # parallel run
        hc.run_parallel(worker=4)
        print(hc.result.calculated_score)

        # test big data
        data = pd.read_csv(r"./test_data/alarm.csv")
        hc = HC(data)
        hc.run_parallel(worker=4, restart=1)
        print(hc.result.calculated_score)
        hc.run()
        print(hc.result.calculated_score)

    def test_hc_restart(self):
        data = pd.read_excel(r"../../datasets/tests/sample0.xlsx")
        dag = DAG()
        score_of_empty_DAG = dag.score(BIC_score, data)
        hc = HC(data)
        hc.run(restart=5, explore_num=2)
        print(hc.result.edges)
        self.assertTrue(hc.result.score(BIC_score, data) >= score_of_empty_DAG)

    def test_ga(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        bic = BIC_score(data)
        dag = DAG()
        score_of_empty_DAG = dag.score(bic)
        ga = GA(data)
        ga.run(max_iter=50)
        print(ga.result.edges)

    def test_kbnl(self):
        data = pd.read_csv(r"./test_data/Asia.csv")
        kbnl = KBNL(data, [r"./test_data/asia_expert0.csv", r"./test_data/asia_expert1.csv"], [0.3, 0.7])
        kbnl.run(restart=5)
        print(kbnl.result.edges, kbnl.result.calculated_score)
        kbnl.run_parallel(worker=4, restart=5)
        print(kbnl.result.edges, kbnl.result.calculated_score)

    def test_k2(self):
        data = pd.read_excel(r"../../datasets/tests/sample0.xlsx")
        dag = DAG()
        score_of_empty_DAG = dag.score(BIC_score, data)
        k = K2(data)
        k.run()
        print(k.result.edges)
