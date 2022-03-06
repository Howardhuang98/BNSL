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
        self.genetic = Genetic(self.asia_data, w=0.5,max_iter=3)

    def test_local_optimizer(self):
        self.genetic.local_optimizer()

    def test_mutate(self):
        print(self.genetic.X)
        self.genetic.mutate()
        print(self.genetic.X)

    def test_update_manager_list(self):
        self.genetic.update_manager_list()
        print(self.genetic.manager_list)

    def test_select_parents(self):
        self.genetic.update_manager_list()
        selected_list = self.genetic.select_parents(3)
        print(selected_list)

    def test_produce_children(self):
        self.genetic.update_manager_list()
        selected_list = self.genetic.select_parents(3)
        children = self.genetic.produce_children(selected_list)
        print(children.shape)

    def test_run(self):
        dag = self.genetic.run()
        print(dag.edges)
        dag.show()
