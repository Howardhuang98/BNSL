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

from dlbn.expert import Expert


class Test_Expert(unittest.TestCase):

    def test_expert(self):
        a = pd.read_excel(r"../../dlbn/tests/test_result.xlsx", index_col=0)
        b = pd.read_excel(r"../../dlbn/tests/test_expert.xlsx", index_col=0)
        e = Expert([a, b], [0.3, 0.7])
        print(e.think('dysp', 'xray'))
        print(e.fused_matrix)

    def test_read_excel(self):
        e = Expert.read_excel([r"test_result.xlsx", r"test_expert.xlsx"], [0.3, 0.7])
        print(e.fused_matrix)
