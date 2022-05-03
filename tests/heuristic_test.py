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
from bnsl.heuristic import HillClimb,Genetic


class Test_Heuristic(unittest.TestCase):

    def setUp(self):
        self.dataset = bnsl.Dataset('alarm')
        self.data = self.dataset.data
        self.bic = bnsl.BIC_score(self.data)

    def test_hc(self):
        hc = HillClimb(self.data, self.bic, max_iter=5, restart=3, num_explore=5)
        hc.climb()
        for h in hc.history:
            print(h)

    def test_ga(self):
        ga = Genetic(self.bic,population=10)
        ga.evolution()
