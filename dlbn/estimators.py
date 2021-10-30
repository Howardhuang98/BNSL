#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
from datetime import datetime
from base import Estimator

from dlbn.heuristic import HillClimb
from dlbn.order_graph import *
from dlbn.score import *

"""
estimator class is used to structure learning with one step, thus it concludes all the workflow
"""


class SPP(Estimator):
    """
    dynamic program estimator: shortest path perspective
    """

    def __init__(self, data):
        self.load_data(data)
        self.result_dag = None
        self.og = None
        # print estimator information
        self.show_est()

    def run(self, score_method:Score):
        variables = list(self.data.columns)
        self.og = OrderGraph(variables)
        self.og.generate_order_graph()
        self.og.add_cost(score_method, self.data)
        self.og.find_shortest_path()
        self.result_dag = self.og.optimal_result()

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



class HC(Estimator):
    """
    greedy hill climb
    """

    def __init__(self, data, io: str = None):
        self.load_data(data)
        self.result_dag = None

    def run(self, **kwargs):
        hc = HillClimb(self.data, **kwargs)
        self.result_dag = hc.climb(**kwargs)
        return self.result_dag



if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    est = SPP(data)
    est.run(MDL_score)
    est.show()
