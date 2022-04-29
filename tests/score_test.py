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
from bnsl.expert import Expert
import bnsl


class Test_Score(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv(r"test_data/Asia.csv")
        dag = bnsl.DAG()
        self.Asian = dag.read(r"test_data/test_save_edge_list.csv")

    def test_mdl(self):
        mdl = bnsl.MDL_score(self.data)
        print("mdl:")
        print(mdl.local_score("asia", ("tub",)))
        print(mdl.local_score("asia", ()))

    def test_l2c(self):
        e1 = pd.read_csv(r"test_data/asia_expert0.csv", index_col=0)
        e2 = pd.read_csv(r"test_data/asia_expert1.csv", index_col=0)
        # e1 = (e1 + 1 / 3) / 2
        # e2 = (e2 + 1 / 3) / 2
        k = e1 * 0.5 + e2 * 0.5
        l2c = bnsl.L2C_score(self.data, k)
        # mdl tub-->asia 265.13
        print("l2c")
        print(l2c.local_score("asia", ("tub",)))
        print(l2c.local_score("asia", ()))

    def test_kfscore(self):
        e1 = pd.read_csv(r"test_data/asia_expert0.csv", index_col=0)
        e2 = pd.read_csv(r"test_data/asia_expert1.csv", index_col=0)
        expert = Expert([e1,e2],[0.5,0.5])
        kfs = bnsl.Knowledge_fused_score(self.data,expert)
        print("kfs")
        print(kfs.local_score("asia", ("tub",)))
        print(kfs.local_score("asia", ()))