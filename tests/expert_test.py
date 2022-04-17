#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/31 10:58  
------------      
"""
import unittest

from bnsl.expert import Expert
from bnsl.graph import DAG


class Test_Expert(unittest.TestCase):
    def setUp(self) -> None:
        self.g = DAG()
        self.g.read("./test_data/Asian_net.csv")
        self.truth = self.g.adj_DataFrame()

    def test_read(self):
        e = Expert([self.truth], [1])
        print(e.accuracy(self.g))
        self.truth = self.truth.applymap(lambda x: 0)
        print(self.truth)
        e = Expert([self.truth], [1])
        print(e.accuracy(self.g))
