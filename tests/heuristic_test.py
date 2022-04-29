#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/31 10:58  
------------      
"""
import unittest

import pandas as pd

import bnsl
from bnsl.heuristic import HillClimb


class Test_HillClimb(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv(r"./test_data/Asia.csv")
        self.bic = bnsl.BIC_score(self.data)

    def test_climb(self):
        hc = HillClimb(self.data, self.bic, restart=3, explore_num=5)
        hc.climb()
        for h in hc.history:
            print(h)
