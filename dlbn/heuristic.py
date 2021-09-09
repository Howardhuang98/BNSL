#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   heuristic.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/9 11:03  
------------      
"""
import logging
from itertools import permutations

import networkx as nx
import pandas as pd
from tqdm import tqdm

from dlbn.direct_graph import DAG
from dlbn.score import Score

logging.basicConfig(filename='run.log', level=logging.INFO)


class HillClimb:
    def __init__(self, data: pd.DataFrame, score_method: Score, dag: DAG = None, **Kwargs):
        self.score_method = score_method
        if not dag:
            self.dag = DAG()
        self.data = data
        self.vars = list(data.columns.values)
        self.dag.add_nodes_from(self.vars)
        self.score_method = score_method(self.data)
        self.tabu_list = []

    def legal_operation(self, current_dag: DAG):
        potential_new_edges = (set(permutations(self.vars, 2)) - set(current_dag.edges()) - set(
            [(v, u) for (u, v) in current_dag.edges()]))

        for u, v in potential_new_edges:
            if not nx.has_path(current_dag, v, u):
                operation = ('+', (u, v))
                if operation not in self.tabu_list:
                    old_parents = list(current_dag.predecessors(v))
                    new_parents = old_parents + [u]
                    score_delta = self.score_method.local_score(v, new_parents) - self.score_method.local_score(v,
                                                                                                                old_parents)
                    yield operation, score_delta

        for u, v in current_dag.edges:
            operation = ('-', (u, v))
            if operation not in self.tabu_list:
                old_parents = list(current_dag.predecessors(v))
                new_parents = old_parents[:]
                new_parents.remove(u)
                score_delta = self.score_method.local_score(v, new_parents) - self.score_method.local_score(v,
                                                                                                            old_parents)
                yield operation, score_delta

        for u, v in current_dag.edges:
            if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(current_dag, u, v))):
                operation = ('flip', (u, v))
                if operation not in self.tabu_list:
                    old_v_parents = list(current_dag.predecessors(v))
                    old_u_parents = list(current_dag.predecessors(u))
                    new_u_parents = old_u_parents + [v]
                    new_v_parents = old_v_parents[:]
                    new_v_parents.remove(u)
                    score_delta = (self.score_method.local_score(v, new_v_parents) + self.score_method.local_score(u,
                                                                                                                   new_u_parents) - self.score_method.local_score(
                        v, old_v_parents) - self.score_method.local_score(u, old_u_parents))
                    yield operation, score_delta

    def climb(self, num_iteration=1000, direction = 'up',**kwargs):
        current = self.dag
        for _ in tqdm(range(num_iteration), desc="Hill climbing"):
            if direction == 'up':
                try:
                    best_operation, score_delta = max(self.legal_operation(current), key=lambda x: x[1])
                except:
                    break
                if score_delta < 0:
                    break
            if direction == 'down':
                try:
                    best_operation, score_delta = min(self.legal_operation(current), key=lambda x: x[1])
                except:
                    break
                if score_delta > 0:
                    break

            logging.info("{}:best operation:{}, score_delta: {}".format(_, best_operation, score_delta))
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
        return self.dag
