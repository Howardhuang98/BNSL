#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   graph_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/30 18:32  
------------      
"""
import unittest

from dlbn.graph import DAG


class Test_graph(unittest.TestCase):

    def test_to_excel_DataFrame(self):
        dag = DAG()
        dag.read(r"../../datasets/asian/Asian net.xlsx")
        dag.to_excel_DataFrame("test_result.csv")

    def test_read_DataFrame_adjacency(self):
        dag = DAG()
        dag.read_DataFrame_adjacency("test_result.csv")
        self.assertEqual(len(dag.edges), 8)
