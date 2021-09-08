#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
import pandas as pd

from dlbn.order_graph import *
from dlbn.score import *

"""
estimator class is used to structure learning with one step, thus it concludes all the workflow
"""


class estimator:
    """
    dynamic program estimator: shortest path perspective
    """

    def __init__(self, data: pd.DataFrame, io: str = None):
        self.data = data
        self.result_dag = None
        self.io = io

    def run(self):
        variables = list(self.data.columns)
        og = OrderGraph(variables)
        og.generate_order_graph()
        og.add_cost(MDL_score, self.data)
        og.find_shortest_path()
        self.result_dag = og.optimal_result(self.io)

        return self.result_dag

if __name__ == '__main__':
    est = estimator(pd.read_csv(r"datasets/Asian.csv"))
    est.run()
