#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   score_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/5 20:53  
------------      
"""
import unittest

import pandas as pd

from dlbn.expert import Expert
from dlbn.score import Knowledge_fused_score, BIC_score, MDL_score,K2_score


class Test_Expert(unittest.TestCase):

    def test_state_count0(self):
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        bic = BIC_score(data)
        table = bic.state_count('tub', ['asia'])
        print(table)

    def test_state_count1(self):
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        bic = BIC_score(data)
        table = bic.state_count('tub', [])
        print(table)

    def test_mdl(self):
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        mdl = MDL_score(data)
        ls0 = mdl.local_score('tub', [])
        ls1 = mdl.local_score('tub', ['asia'])
        print(ls0,ls1)


    def test_kfs(self):
        a = pd.read_excel(r"../../dlbn/test/test_result.xlsx", index_col=0)
        b = pd.read_excel(r"../../dlbn/test/test_expert.xlsx", index_col=0)
        e = Expert([a, b], [1, 0])
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        k = Knowledge_fused_score(data, e)
        bic =BIC_score(data)
        print(e.fused_matrix)
        print(k.local_score('tub',['asia']))
        print(k.local_score('tub', ['asia', 'either']))
        print(bic.local_score('tub', ['asia']))
        print(bic.local_score('tub', ['asia','either']))



