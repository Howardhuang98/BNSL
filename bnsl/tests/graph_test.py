#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   graph_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/30 18:32  
------------      
"""
import unittest

import pandas as pd

from bnsl.graph import DAG
from bnsl.score import BIC_score


class Test_graph(unittest.TestCase):

    def setUp(self):
        dag = DAG()
        self.asia_net = dag.read(r"../../datasets/asian/Asian_net.csv")

    def test_to_excel_DataFrame(self):
        dag = DAG()
        dag.read(r"../../datasets/asian/Asian net.xlsx")
        dag.to_excel_DataFrame("test_result.csv")

    def test_read_DataFrame_adjacency(self):
        dag = DAG()
        dag.read_DataFrame_adjacency("test_result.csv")
        self.assertEqual(len(dag.edges), 8)

    def test_score(self):
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        bic = BIC_score(data)
        print(self.asia_net.score(bic))



