#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   graph_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/30 18:32  
------------      
"""
import copy
import unittest
from bnsl.graph import DAG, compare
from bnsl.score import BIC_score


class Test_graph(unittest.TestCase):

    def setUp(self):
        self.Asian = DAG()
        self.Asian.read(r"../datasets/asian/Asian_net.csv")
        self.Asian.summary()

    def test_save(self):
        self.Asian.save(r"./test_data/test_save_edge_list.csv")
        self.Asian.save(r"./test_data/test_save_adjacent_matrix.csv", mode='adjacent_matrix')

    def test_read(self):
        g = DAG()
        g.read(r"./test_data/test_save_adjacent_matrix.csv", mode='adjacent_matrix')
        g.summary()

    def test_compare(self):
        other = copy.deepcopy(self.Asian)
        other.remove_edge('asia', 'tub')
        metrics = compare(self.Asian, other)
        print(metrics)

