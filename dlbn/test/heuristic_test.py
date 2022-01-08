#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic_test.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/31 10:58  
------------      
"""
import unittest

import pandas as pd

from dlbn.graph import DAG
from dlbn.heuristic import HillClimb


class Test_HillClimb(unittest.TestCase):

    def test_possible_operation(self):
        data = pd.read_excel(r"../../datasets/test/sample0.xlsx")
        hc = HillClimb(data)
        po = [i for i in hc.possible_operation(['A'])]
        print(po)
        print(len(po))

    def test_possible_operation1(self):
        dag = DAG()
        dag.read_DataFrame_adjacency("test_result.xlsx")
        data = pd.read_csv(r"../../datasets/asian/Asian.csv")
        hc = HillClimb(data,initial_dag=dag)
        po = [i for i in hc.possible_operation(['tub'])]
        print(po)
        print(len(po))
