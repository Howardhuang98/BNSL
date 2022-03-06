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
        self.genetic = Genetic(self.asia_data, w=0.5)

    def test_local_optimizer(self):
        self.genetic.local_optimizer()

    def test_mutate(self):
        print(self.genetic.X)
        self.genetic.mutate()
        print(self.genetic.X)
