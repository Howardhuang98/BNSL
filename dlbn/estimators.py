#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
from datetime import datetime

import matplotlib.pyplot as plt

from dlbn.base import Estimator
from dlbn.bionics import Genetic
from dlbn.graph import *
from dlbn.heuristic import HillClimb, SimulatedAnnealing
from dlbn.score import *
from dlbn.pc import *

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

    def __init__(self, data, score_method, **kwargs):
        self.load_data(data)
        self.result_dag = None
        self.show_est()
        self.score_method = score_method(self.data, **kwargs)

    def run(self, **kwargs):
        hc = HillClimb(self.data, self.score_method)
        self.result_dag = hc.climb(**kwargs)
        return self.result_dag


class SA(Estimator):
    """
    
    """

    def __init__(self, data, score_method: BIC_score, **kwargs):
        self.load_data(data)
        self.result_dag = None
        self.show_est()
        self.score_method = score_method(data, **kwargs)

    def run(self):
        sa = SimulatedAnnealing(self.data, self.score_method)
        self.result_dag = sa.run()
        return self.result_dag


class PC(Estimator):
    def __init__(self, data):
        """

        :param data:
        """
        self.load_data(data)
        self.result_dag = None
        self.show_est()

    def run(self):
        skl, sep_set = estimate_skeleton(self.data)
        cpdag = estimate_cpdag(skl, sep_set)
        cpdag = nx.relabel.relabel_nodes(cpdag, dict(zip(range(len(data.columns)), data.columns)))
        return cpdag

class GA(Estimator):
    def __init__(self,data):
        self.load_data(data)
        self.result_dag = None

    def run(self,score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5, w=0.05):
        pso = Genetic(self.data, score_method=BIC_score, pop=pop, max_iter=max_iter, c1=c1, c2=c2, w=w)
        solu, history = pso.run()
        g = DAG()
        g.from_genome(solu,self.data.columns)
        return g


if __name__ == '__main__':
    # data = pd.read_csv(r"../datasets/Asian.csv")
    # expert_data = pd.read_csv(r"../datasets/Asian expert.csv", index_col=0)
    # expert = Expert(expert_data)
    # est = SA(data, score_method=Knowledge_fused_score, expert=expert)
    # est.run()
    # est.show()
    # data = pd.read_csv(r"../datasets/Asian.csv", )
    # data[data == 'no'] = 0
    # data[data == 'yes'] = 1
    # data = data.astype(int)
    # est = PC(data)
    # est.run()
    # est.show()

    # data = pd.read_csv(r"../datasets/Asian.csv", )
    # data[data == 'no'] = 0
    # data[data == 'yes'] = 1
    # data = data.astype(int)
    # ground_truth = DAG()
    # ground_truth.read_excel(r"../datasets/Asian net.xlsx")
    # # expert = Expert(path=r"../experiments/expert-100-4.csv")
    # # random_expert = Expert.random_init(data)
    # hc_est = HC(data, BDeu_score)
    # hc_est.run()
    # hc_est.show()
    # print(hc_est.result_dag-ground_truth)
    data = pd.read_csv(r"../datasets/Asian.csv")
    est = GA(data)
    dag = est.run(max_iter=150)
    nx.draw_networkx(dag)
    dag.to_excel()

