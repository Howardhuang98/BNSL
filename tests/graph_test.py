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
import bnsl
from bnsl.graph import compare, random_dag


class Test_graph(unittest.TestCase):

    def setUp(self):
        self.Asian = bnsl.DAG()
        self.Asian.read(r"../datasets/asian/Asian_net.csv")
        self.Asian.summary()

    def test_save(self):
        self.Asian.save(r"./test_data/test_save_edge_list.csv")
        self.Asian.save(r"./test_data/test_save_adjacent_matrix.csv", mode='adjacent_matrix')

    def test_read(self):
        g = bnsl.DAG()
        g.read(r"./test_data/test_save_adjacent_matrix.csv", mode='adjacent_matrix')
        g.summary()

    def test_compare_and_random_dag(self):
        other = copy.deepcopy(self.Asian)
        other.remove_edge('asia', 'tub')
        metrics = compare(self.Asian, other)
        print(metrics)
        node_list = list(self.Asian.nodes)
        dag = random_dag(node_list)
        metrics = compare(self.Asian, dag)
        print(metrics)
