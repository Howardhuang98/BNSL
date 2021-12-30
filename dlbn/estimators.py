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
from dlbn.score import BIC_score, MDL_score


class DP(Estimator):
    """ Dynamic program estimator class
    reference: 《Learning Optimal Bayesian Networks: A Shortest Path Perspective》
    :param: data, np.array or pd.Dataframe
    """

    def __init__(self, data):
        super(DP, self).__init__()
        self.load_data(data)

    def run(self, score_method=MDL_score):
        """
        run the dynamic program estimator, an exact algorithm. MDL score is used as the score criteria, it return the
        dag with minimum score.
        :param score_method: MDL score
        :return: the dag with minimum score
        """
        pg = generate_parent_graph(self.data, score_method)
        og = generate_order_graph(self.data, pg)
        self.result = order2dag(og, self.data)
        return self.result


class HC(Estimator):
    """
    Greedy hill climb estimator
    """

    def __init__(self, data):
        super(HC, self).__init__()
        self.load_data(data)

    def run(self, score_method=BIC_score, direction='up', initial_dag=None, max_iter=10000, restart=1):
        """
        run the HC estimator.
        :param score_method: score method, usually select BIC score or BDeu score
        :param direction:  try to find the maximum of minimum score
        :param initial_dag: the initial dag
        :param max_iter: the number of maximum iteration
        :param restart: the number of restart times, every restart will random initialize a start DAG
        :return: an approximate maximum or minimum scored DAG
        """
        hc = HillClimb(self.data, score_method, initial_dag=initial_dag, max_iter=max_iter,
                       restart=restart)
        self.result = hc.climb(direction)
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
    """
    Genetic algorithm estimator class
    """
    def __init__(self, data):
        super(GA, self).__init__()
        self.load_data(data)

    def run(self, score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5, w=0.05):
        """
        run the genetic algorithm estimator
        :param score_method: score criteria
        :param pop: number of population
        :param max_iter: maximum iteration number
        :param c1: [0,1] the probability of crossover with personal historical best genome
        :param c2: [0,1] the probability of crossover with global historical best genome
        :param w: the probability of mutation
        :return: the dag with maximum score
        """
        ga = Genetic(self.data, score_method=BIC_score, pop=pop, max_iter=max_iter, c1=c1, c2=c2, w=w)
        solu, history = ga.run()
        self.result.from_genome(solu, self.data.columns)
        return self.result


if __name__ == '__main__':
    pass
