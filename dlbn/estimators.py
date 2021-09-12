#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from dlbn.heuristic import HillClimb
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
        self.result_dag = og.optimal_result()

        return self.result_dag

    def save(self, io: str = None):
        self.io = io
        if not self.result_dag:
            raise ValueError("Please run! there is no result dag")
        elif not io:
            now = datetime.now()
            self.io = "{}-{}-{}-{}-{}.csv".format(now.year, now.month, now.day, now.hour, now.second)

        df = nx.to_pandas_edgelist(self.result_dag)
        df.to_csv(self.io)

        return None

    def show(self, score_method: Score = MDL_score):
        nx.draw_networkx(self.result_dag)
        plt.title("Bayesian network with Score={}".format(self.result_dag.score(score_method, self.data)))
        plt.show()
        return None


class HC_estimator(estimator):
    """
    greedy hill climb
    """

    def __init__(self, data: pd.DataFrame, io: str = None):
        super(HC_estimator, self).__init__(data, io)

    def run(self, **kwargs):
        hc = HillClimb(self.data, **kwargs)
        self.result_dag = hc.climb(**kwargs)

        return self.result_dag


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    est = HC_estimator(data)
    est.run(num_iteration=100, score_method=MDL_score, direction='down')
    print(est.result_dag.score(score_method=MDL_score,data=data))
    est.show()
    asia_net = DAG()
    asia_net.read_excel(r"../datasets/Asian net.xlsx")
    shd = est.result_dag - asia_net
    print(shd)
    est2 = estimator(data)
    est2.run()
    est2.show()
    shd2 = est2.result_dag - asia_net
    print(shd2)
    print("原网络评分{}".format(asia_net.score(score_method=MDL_score,data=data)))
