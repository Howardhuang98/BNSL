#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/9 11:03  
------------      
"""
import random
import sys
import warnings
from copy import deepcopy
from itertools import product
from math import exp

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .graph import DAG
from .score import Score, BIC_score
from .graph import random_dag
from .log import bnsl_log

class HillClimb:
    """
    Hill climb search algorithm.

    Args:
        data: pd.DataFrame: observed data.
        Score_method: Score: score method instance.
        initial_dag: DAG: initial dag.
        max_iter: int: maximum iteration times.
        restart: int: number of restart.
        num_explore: int: number of explored nodes.
        num_parents: int: number of parents.
    """

    def __init__(self, data: pd.DataFrame, Score_method: Score = BIC_score, initial_dag: DAG = None, max_iter=100,
                 restart=1, num_explore=3, num_parents=5):
        self.data = data
        self.nodes = list(data.columns.values)
        self.n_samples = len(data)

        if not isinstance(Score_method, Score):
            raise ValueError("Score method has to be Score instance")
        self.s = Score_method

        if initial_dag:
            self.dag = initial_dag
        else:
            self.dag = random_dag(self.nodes, num_parents)
        self.tabu_list = []
        self.max_iter = max_iter
        self.restart = restart
        self.explore_num = num_explore
        if num_explore > len(self.nodes):
            warnings.warn("Explore number exceeds the node number")
        self.num_parents = num_parents
        self.history = []

    def possible_operation(self, dag, node_list):
        """
        iterator, yield possible operation for a node list.
        :param: node_list, the node list will be explored
        :return: possible operation for one node
        """
        for node in node_list:
            potential_new_edges = set(product([node], dag.nodes)) | set(product(dag.nodes, [node]))
            potential_new_edges -= {(node, node)}
            for u, v in potential_new_edges:
                if (u, v) in [(a, b) for a, b in dag.edges]:
                    operation = ('-', (u, v))
                    if operation not in self.tabu_list:
                        old_parents = list(dag.predecessors(v))
                        new_parents = old_parents[:]
                        new_parents.remove(u)
                        score_delta = self.s.local_score(v, tuple(new_parents)) - self.s.local_score(v,
                                                                                                     tuple(old_parents))
                        yield operation, score_delta

                    if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(dag, u, v))):
                        operation = ('flip', (u, v))
                        if operation not in self.tabu_list:
                            old_v_parents = list(dag.predecessors(v))
                            old_u_parents = list(dag.predecessors(u))
                            new_u_parents = old_u_parents + [v]
                            new_v_parents = old_v_parents[:]
                            new_v_parents.remove(u)
                            score_delta = (self.s.local_score(v, tuple(new_v_parents)) + self.s.local_score(u,
                                                                                                            tuple(
                                                                                                                new_u_parents)) - self.s.local_score(
                                v, tuple(old_v_parents)) - self.s.local_score(u, tuple(old_u_parents)))
                            yield operation, score_delta
                else:
                    if not nx.has_path(dag, v, u):
                        operation = ('+', (u, v))
                        if operation not in self.tabu_list:
                            old_parents = list(dag.predecessors(v))
                            new_parents = old_parents + [u]
                            score_delta = self.s.local_score(v, tuple(new_parents)) - self.s.local_score(v,
                                                                                                         tuple(
                                                                                                             old_parents))
                            yield operation, score_delta

    def climb(self, direction='up'):
        """
        execute hill climb search
        :param direction: the direction of search, up or down
        :return: a DAG instance
        """

        result = []
        for i in range(self.restart):
            if i != 0:
                temp = random_dag(self.nodes, self.num_parents)
            else:
                temp = deepcopy(self.dag)
            scores = [temp.score(self.s)]
            for _ in range(self.max_iter):
                # randomly select a node list
                node_list = random.choices(self.nodes, k=self.explore_num)
                sys.stdout.write(f"\r {_}-th hill climbing")
                sys.stdout.flush()
                if direction == 'up':
                    sys.stdout.write(f"\r {_}-th hill climbing: exploring {node_list}")
                    sys.stdout.flush()
                    best_operation, score_delta = max(self.possible_operation(temp, node_list), key=lambda x: x[1])
                    if score_delta < 0:
                        break
                else:
                    best_operation, score_delta = min(self.possible_operation(temp, node_list), key=lambda x: x[1])
                    if score_delta > 0:
                        break
                bnsl_log.info(f"{_}-th iteration: best operation: {best_operation}, score delta: {score_delta}")
                if best_operation[0] == '+':
                    temp.add_edge(*best_operation[1])
                if best_operation[0] == '-':
                    self.tabu_list.append(('+', best_operation[1]))
                    temp.remove_edge(*best_operation[1])
                if best_operation[0] == 'flip':
                    self.tabu_list.append(('flip', best_operation[1]))
                    u, v = best_operation[1]
                    temp.remove_edge(u, v)
                    temp.add_edge(v, u)
                scores.append(scores[-1] + score_delta)
            self.history.append(scores)
            temp_score = temp.score(self.s)
            result.append((temp, temp_score))
            print(f"\r {i}-th restarting found dag with score: {temp_score}")

        if direction == 'up':
            self.dag = max(result, key=lambda x: x[1])[0]
        if direction == 'down':
            self.dag = min(result, key=lambda x: x[1])[0]
        return self.dag


class SimulatedAnnealing:
    def __init__(self, data: pd.DataFrame, score_method, dag: DAG = None):
        self.data = data
        self.score_method = score_method
        if dag is None:
            self.dag = DAG()
            self.dag.add_nodes_from(list(data.columns.values))

    def run(self, T=1000.0, k=0.99, num_iteration=1000):
        current_dag = self.dag
        for _ in tqdm(range(num_iteration)):
            legal_operations = list(current_dag.legal_operations())
            operation = random.sample(legal_operations, 1)[0]
            score_delta = current_dag.score_delta(operation, self.score_method)
            if score_delta > 0:
                current_dag.do_operation(operation)
            if score_delta < 0:
                if exp(score_delta / T) > random.uniform(0, 1):
                    current_dag.do_operation(operation)
            # cooling
            T *= k
        self.dag = current_dag
        return self.dag


if __name__ == '__main__':
    pass
