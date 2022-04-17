#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   k2_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/17 20:00  
------------      
"""
import unittest

import pandas as pd
from bnsl import k2
from bnsl.score import BIC_score


class Test_Expert(unittest.TestCase):

    def test1(self):
        data = pd.read_csv(r"./test_data/Asia.csv")
        bic = BIC_score(data)
        g = k2.order_to_dag(list(data.columns),3,bic)
        print(g.edges)