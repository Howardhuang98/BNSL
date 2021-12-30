#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
from dlbn.base import Estimator
from dlbn.bionics import Genetic
from dlbn.dp import generate_order_graph, generate_parent_graph, order2dag
from dlbn.heuristic import HillClimb, SimulatedAnnealing
from dlbn.pc import *
from dlbn.score import *


class DP(Estimator):
    """

    """

    def __init__(self, data):
        super(DP, self).__init__()
        self.load_data(data)

    def run(self, score_method=MDL_score):
        pg = generate_parent_graph(self.data, score_method)
        og = generate_order_graph(self.data, pg)
        self.result = order2dag(og, self.data)
        return self.result


class HC(Estimator):
    """
    greedy hill climb
    """

    def __init__(self, data, score_method, **kwargs):
        super(HC, self).__init__()
        self.load_data(data)
        self.show_est()
        self.score_method = score_method(self.data, **kwargs)

    def run(self, **kwargs):
        hc = HillClimb(self.data, self.score_method)
        self.result = hc.climb(**kwargs)
        return self.result


class SA(Estimator):
    """
    
    """

    def __init__(self, data, score_method=BIC_score, **kwargs):
        super(SA, self).__init__()
        self.load_data(data)
        self.show_est()
        self.score_method = score_method(data)

    def run(self):
        sa = SimulatedAnnealing(self.data, self.score_method)
        self.result = sa.run()
        return self.result


class PC(Estimator):
    def __init__(self, data):
        """

        :param data:
        """
        super(PC, self).__init__()
        self.load_data(data)
        self.show_est()

    def run(self):
        skl, sep_set = estimate_skeleton(self.data)
        cpdag = estimate_cpdag(skl, sep_set)
        cpdag = nx.relabel.relabel_nodes(cpdag, dict(zip(range(len(data.columns)), data.columns)))
        self.result = cpdag
        return self.result


class GA(Estimator):
    def __init__(self, data):
        super(GA, self).__init__()
        self.load_data(data)

    def run(self, score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5, w=0.05):
        ga = Genetic(self.data, score_method=BIC_score, pop=pop, max_iter=max_iter, c1=c1, c2=c2, w=w)
        solu, history = ga.run()
        self.result.from_genome(solu, self.data.columns)
        return self.result


if __name__ == '__main__':
    pass
