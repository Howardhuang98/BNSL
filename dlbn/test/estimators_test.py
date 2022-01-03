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

from dlbn.estimators import DP, HC, GA
from dlbn.graph import DAG
from dlbn.score import MDL_score, BIC_score


class Test_estimator(unittest.TestCase):
    """
    sample0.xlsx has the same value for E, F variable, thus there should be an edge between E and F, but
    unable to determine the direction.
    """

    def test_dp(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        dag = DAG()
        score = dag.score(MDL_score, data)
        dp = DP(data)
        dp.run()
        print(dp.result.edges)
        self.assertTrue(dp.result.score(MDL_score, data) <= score)

    def test_hc(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        dag = DAG()
        score_of_empty_DAG = dag.score(BIC_score, data)
        hc = HC(data)
        hc.run()
        print(hc.result.edges)
        self.assertTrue(hc.result.score(BIC_score, data) >= score_of_empty_DAG)

    def test_hc_restart(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        dag = DAG()
        score_of_empty_DAG = dag.score(BIC_score, data)
        hc = HC(data)
        hc.run(restart=5,explore_num=2)
        print(hc.result.edges)
        self.assertTrue(hc.result.score(BIC_score, data) >= score_of_empty_DAG)

    def test_ga(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        dag = DAG()
        score_of_empty_DAG = dag.score(BIC_score, data)
        ga = GA(data)
        ga.run(max_iter=50)
        print(ga.result.edges)
        self.assertTrue(ga.result.score(BIC_score, data) >= score_of_empty_DAG)

