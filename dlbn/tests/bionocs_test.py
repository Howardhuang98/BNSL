#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bionocs_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/3/6 19:27  
------------      
"""
import unittest

import pandas as pd

from dlbn.graph import DAG
from dlbn.score import BIC_score
from dlbn.bionics import Genetic


class Test_genetic(unittest.TestCase):

    def setUp(self):
        dag = DAG()
        self.asia_net = dag.read(r"../../datasets/asian/Asian_net.csv")
        self.asia_data = pd.read_csv(r"../../datasets/asian/Asian.csv")

    def test_run(self):
        bic = BIC_score(self.asia_data)
        print("goal score", self.asia_net.score(bic))
        g = Genetic(self.asia_data, max_iter=100, patience=50, pop=5)
        dag = g.run()
        print(g.history)
        print(dag.edges)
        print(self.asia_net - dag)

    def test_run_big_data(self):
        hg = DAG()
        h_net = hg.read(r"../../datasets/hailfinder/hailfinder_net.csv")
        h_data = pd.read_csv(r"../../datasets/hailfinder/hailfinder.csv", index_col=0)[:5000]
        bic = BIC_score(h_data)
        print(h_net.score(bic))
        g = Genetic(h_data, pop=3, max_iter=100)
        dag = g.run()
        print(dag.edges)
        print(h_net - dag)
        print(g.history)
