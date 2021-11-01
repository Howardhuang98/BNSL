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
from dlbn.expert import Expert
from dlbn.graph import *
from dlbn.heuristic import HillClimb, SimulatedAnnealing
from dlbn.score import *

"""
estimators

score based estimator work flow:
load data
show itself
instance a score method
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

    def run(self, score_method: Score = MDL_score):
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

    def __init__(self, data, score_method,**kwargs):
        self.load_data(data)
        self.result_dag = None
        self.show_est()
        self.score_method = score_method(self.data,**kwargs)

    def run(self, **kwargs):
        hc = HillClimb(self.data, self.score_method)
        self.result_dag = hc.climb(**kwargs)
        return self.result_dag


class SA(Estimator):
    """
    
    """

    def __init__(self, data, score_method, **kwargs):
        self.load_data(data)
        self.result_dag = None
        self.show_est()
        self.score_method = score_method(data, **kwargs)

    def run(self):
        sa = SimulatedAnnealing(self.data, self.score_method)
        self.result_dag = sa.run()
        return self.result_dag






if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    expert_data = pd.read_csv(r"../datasets/Asian expert.csv", index_col=0)
    expert = Expert(expert_data)
    est = SA(data,score_method=Knowledge_fused_score,expert=expert)
    est.run()
    est.show()
