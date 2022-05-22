#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   estimators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/8 14:21  
------------      
"""
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd

import bnsl
from .base import Estimator
from .dp import generate_order_graph, generate_parent_graph, order2dag
from .expert import Expert
from .heuristic import HillClimb, SimulatedAnnealing, Genetic
from .pc import *
from .score import BIC_score, MDL_score, Knowledge_fused_score
from .k2 import order_to_dag


class HC(Estimator):
    """
    Greedy hill climb estimator.
    """

    def __init__(self, data):
        super(HC, self).__init__()
        self.load_data(data)

    def run(self, score_method=BIC_score, direction='up', initial_dag=None, max_iter=10000, restart=1, explore_num=5,
            **kwargs):
        """
        run the HC estimator.

        For example:

        .. code-block:: python

            hc = HC(data)
            dag = hc.run()


        Args:
            score_method: score function.
            direction: the climb direction.
            initial_dag: the begin dag.
            max_iter: the maximum iteration number.
            restart: the restart number of hill climb.
            explore_num: in every iteration, estimation will explore the number of node to find possible operations.
            **kwargs:

        Returns:
            A DAG instance.
        """
        s = score_method(self.data, **kwargs)
        hc = HillClimb(self.data, s, initial_dag=initial_dag, max_iter=max_iter,
                       restart=restart, num_explore=explore_num)
        self.result = hc.climb(direction)
        return self.result

    def run_parallel(self, worker=4, **kwargs):
        """
        Run estimate with multi processes.

        Args:
            worker: number of processes.
            **kwargs: check the arguments in ``HC.run()``.

        Returns:
            A DAG instance.
        """
        kwargs["instance"] = self
        arguments = [kwargs for i in range(worker)]
        with Pool(processes=worker) as pool:
            result = pool.map(_process, arguments)
        i = np.argmax([dag.calculated_score for dag in result])
        self.result = result[i]
        return self.result


class DP(Estimator):
    """ Dynamic planning estimator.

    Dynamic planning algorithm is an exact algorithm, might consume long time.
    reference: 《Learning Optimal Bayesian Networks: A Shortest Path Perspective》

    Attributes:
        data: observed data.

    """

    def __init__(self, data):
        """
        Initialize the DP estimator.

        Args:
            data: observed data.
        """
        super(DP, self).__init__()
        self.load_data(data)

    def run(self, score_method=MDL_score):
        """
        Run the dynamic program estimator.

        Args:
            score_method: score function.

        Returns:
            A DAG instance.
        """
        pg = generate_parent_graph(self.data, score_method)
        og = generate_order_graph(self.data, pg)
        self.result = order2dag(og, self.data)
        return self.result


def _process(arguments):
    result = arguments["instance"].run(**arguments)
    return result


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

    def run(self):
        skl, sep_set = estimate_skeleton(self.data)
        cpdag = estimate_cpdag(skl, sep_set)
        cpdag = nx.relabel.relabel_nodes(cpdag, dict(zip(range(len(self.data.columns)), self.data.columns)))
        self.result = cpdag
        return self.result


class GA(Estimator):
    """
    Genetic algorithm estimator.

    Genetic algorithm uses genome to represent DAG, and genomes can crossover and mutate. Every iteration simulates the
    nature revolutionary process to select DAG with highest score.

    References: Larrañaga, P., & Poza, M. (1994). Structure learning of Bayesian networks by genetic algorithms. In
    New Approaches in Classification and Data Analysis (pp. 300-307). Springer, Berlin, Heidelberg.

    Attributes:
        history: the evolutionary history.

    Methods:
        run: run the estimator.


    """

    def __init__(self, data,population=40, c1=0.5, c2=0.5, w=0.05, num_parents=5, max_iter=100):
        super(GA, self).__init__()
        self.load_data(data)
        self.history = None
        self.pop = population
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_parents = num_parents
        self.max_iter = max_iter
        self.result = None

    def run(self):
        """
        run the genetic algorithm estimator.
        Returns:
            A DAG instance or a tuple.
        """
        bic = BIC_score(self.data)
        ga = Genetic(bic, population=self.pop, c1=self.c1, c2=self.c2, w=self.w, num_parents=self.num_parents, max_iter=self.max_iter)
        self.result = ga.evolution()
        return self.result


class KBNL(Estimator):
    """
    KBNL estimator, observed data, expert data and expert confidence are needed to initialize the estimator.
    """

    def __init__(self, data, expert_data: list, expert_confidence: list, ):
        super(KBNL, self).__init__()
        self.load_data(data)
        if isinstance(expert_data[0], pd.DataFrame):
            self.expert = Expert(expert_data, expert_confidence)
        if isinstance(expert_data[0], str):
            self.expert = Expert.read(expert_data, confidence=expert_confidence)

    def run(self, initial_dag=None, max_iter=10000, restart=5, explore_num=5, **kwargs):
        """
        run the KBNL estimator.
        :param initial_dag: the initial dag
        :param max_iter: the number of maximum iteration
        :param restart: the number of restart times, every restart will random initialize a start DAG
        :param explore_num:
        :return: an maximum knowledge fused scored DAG
        """
        s = Knowledge_fused_score(self.data, self.expert)
        hc = HillClimb(self.data, s, initial_dag=initial_dag, max_iter=max_iter,
                       restart=restart, num_explore=explore_num, **kwargs)
        self.result = hc.climb()
        return self.result

    def run_parallel(self, worker=4, **kwargs):
        """
        :return:
        """
        kwargs["instance"] = self
        arguments = [kwargs for i in range(worker)]
        with Pool(processes=worker) as pool:
            result = pool.map(_process, arguments)
        i = np.argmax([dag.calculated_score for dag in result])
        self.result = result[i]
        return self.result


class K2(Estimator):
    def __init__(self, data: pd.DataFrame, score_method=BIC_score):
        super(K2).__init__()
        self.score_method = score_method(data)
        self.order = list(data.columns)

    def run(self):
        self.result = order_to_dag(self.order, 3, self.score_method)
        return self.result


class L2C(Estimator):
    def __init__(self, data: pd.DataFrame, E: List, c: List):
        super(L2C, self).__init__()
        self.load_data(data)
        self.E = E
        self.c = c
        E_primer = []
        for i in range(len(self.E)):
            E_primer.append(self.c[i] * (self.E[i] + 1 / 3) / 2)
        self.K = E_primer[0]
        for u in self.K.index:
            for v in self.K.columns:
                for i in range(1, len(self.E)):
                    self.K.loc[u][v] = (self.K.loc[u][v]+E_primer[i].loc[u][v])/2
        self.l2c_score = bnsl.L2C_score(self.data, self.K)

    def run(self, initial_dag=None, max_iter=10000, restart=1, explore_num=5,
            **kwargs):
        hc = HillClimb(self.data, self.l2c_score, initial_dag=initial_dag, max_iter=max_iter,
                       restart=restart, num_explore=explore_num)
        self.result = hc.climb()
        return self.result
