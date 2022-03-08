#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/9 11:03  
------------      
"""
import random
from copy import deepcopy
from itertools import permutations, product
from math import exp

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from dlbn.graph import DAG
from dlbn.score import Score, BIC_score


class HillClimb:
    """
    Hill climb search class
    """

    def __init__(self, data: pd.DataFrame, Score_method: Score = BIC_score, initial_dag: DAG = None, max_iter=10000,
                 restart=1,explore_num=1, **kwargs):
        self.data = data
        self.Score_method = Score_method
        self.s = self.Score_method(self.data, **kwargs)
        if initial_dag:
            self.dag = initial_dag
        else:
            self.dag = DAG()
        self.vars = list(data.columns.values)
        self.dag.add_nodes_from(self.vars)
        self.tabu_list = []
        self.max_iter = max_iter
        self.restart = restart
        self.explore_num = explore_num
        self.kwargs = kwargs
        self.score_result = None

    def possible_operation(self, node_list=[]):
        """
        iterator, yield possible operation for a node list.
        :param: node_list, the node list will be explored
        :return: possible operation for one node
        """
        for node in node_list:
            potential_new_edges = set(product([node], self.dag.nodes)) | set(product(self.dag.nodes, [node]))
            potential_new_edges -= {(node, node)}
            for u, v in potential_new_edges:
                if (u, v) in [(a, b) for a, b in self.dag.edges]:
                    operation = ('-', (u, v))
                    if operation not in self.tabu_list:
                        old_parents = list(self.dag.predecessors(v))
                        new_parents = old_parents[:]
                        new_parents.remove(u)
                        score_delta = self.s.local_score(v, new_parents) - self.s.local_score(v,
                                                                                              old_parents)
                        yield operation, score_delta

                    if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(self.dag, u, v))):
                        operation = ('flip', (u, v))
                        if operation not in self.tabu_list:
                            old_v_parents = list(self.dag.predecessors(v))
                            old_u_parents = list(self.dag.predecessors(u))
                            new_u_parents = old_u_parents + [v]
                            new_v_parents = old_v_parents[:]
                            new_v_parents.remove(u)
                            score_delta = (self.s.local_score(v, new_v_parents) + self.s.local_score(u,
                                                                                                     new_u_parents) - self.s.local_score(
                                v, old_v_parents) - self.s.local_score(u, old_u_parents))
                            yield operation, score_delta
                else:
                    if not nx.has_path(self.dag, v, u):
                        operation = ('+', (u, v))
                        if operation not in self.tabu_list:
                            old_parents = list(self.dag.predecessors(v))
                            new_parents = old_parents + [u]
                            score_delta = self.s.local_score(v, new_parents) - self.s.local_score(v,
                                                                                                  old_parents)
                            yield operation, score_delta

    def climb(self, direction='up'):
        """
        execute hill climb search
        :param direction: the direction of search, up or down
        :return: a DAG instance
        """
        result = []
        self.score_result = []
        for i in range(self.restart):
            if i != 0:
                self.dag.remove_edges_from([e for e in self.dag.edges])
                self.dag.random_dag()
            for _ in tqdm(range(self.max_iter), desc="Hill climbing"):
                # randomly select a node list
                node_list = random.sample(list(self.dag.nodes),self.explore_num)
                if direction == 'up':
                    best_operation, score_delta = max(self.possible_operation(node_list), key=lambda x: x[1])
                    if score_delta < 0:
                        break
                if direction == 'down':
                    best_operation, score_delta = min(self.possible_operation(node_list), key=lambda x: x[1])
                    if score_delta > 0:
                        break
                if best_operation[0] == '+':
                    self.tabu_list.append(('-', best_operation[1]))
                    self.dag.add_edge(*best_operation[1])
                if best_operation[0] == '-':
                    self.tabu_list.append(('+', best_operation[1]))
                    self.dag.remove_edge(*best_operation[1])
                if best_operation[0] == 'flip':
                    self.tabu_list.append(('flip', best_operation[1]))
                    u, v = best_operation[1]
                    self.dag.remove_edge(u, v)
                    self.dag.add_edge(v, u)
            result.append(deepcopy(self.dag))
            self.score_result.append(self.dag.score(self.s))
        if direction == 'up':
            self.dag = result[np.argmax(self.score_result)]
        if direction == 'down':
            self.dag = result[np.argmin(self.score_result)]
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
