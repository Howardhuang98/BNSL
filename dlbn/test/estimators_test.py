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

from dlbn.estimators import DP
from dlbn.graph import DAG
from dlbn.score import MDL_score


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
