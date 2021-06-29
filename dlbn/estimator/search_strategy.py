#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   search_strategy.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/28 15:30  
------------      
"""
from itertools import permutations

from tqdm import trange

from dlbn.estimator.score_function import *


class SearchStrategy:
    def __init__(self):
        pass

    def run(self):
        pass


class HillClimb(SearchStrategy):
    def __init__(self, score_function, data, max_iter=1000, initial_dag: DAG = None, show_process=True):
        """
        hill climb search
        :param score_function: BicScore,MdlScore
        :param data: pd.Dataframe or Data
        """
        super(HillClimb, self).__init__()
        if isinstance(data, pd.DataFrame):
            self.data = Data(data)
        else:
            self.data = data
        self.score = score_function(self.data.contingency_table())
        self.max_iter = max_iter
        if not initial_dag:
            self.initial_DAG = DAG()
        else:
            self.initial_DAG = initial_dag
        self.initial_DAG.add_nodes_from(self.data.variables)
        self.show_process = show_process

    def legal_operation(self, dag: DAG):
        """
        a generator that generates a list of legal operation with its score difference
        :param dag:
        :param DAG
        :yield (legal operation, score diff)
        """
        # all potential operation for adding edges
        potential_adding_edges = (
                set(permutations(self.data.variables, 2))
                - set(dag.edges())
                - set([(Y, X) for (X, Y) in dag.edges()])
        )
        for (X, Y) in potential_adding_edges:
            operation = ('+', (X, Y))
            # check cycle
            if nx.has_path(dag, Y, X):
                continue
            else:
                try:
                    old_parents = list(dag.predecessors(Y))
                except:
                    old_parents = []
            new_parents = old_parents + [X]
            old_score = self.score.local_score(Y, old_parents)
            new_score = self.score.local_score(Y, new_parents)
            score_diff = new_score - old_score
            yield operation, score_diff

        # all potential operation for deleting edge
        for (X, Y) in dag.edges:
            operation = ('-', (X, Y))
            try:
                old_parents = list(dag.predecessors(Y))
            except nx.NetworkXError:
                old_parents = []
            new_parents = old_parents
            new_parents.remove(X)
            old_score = self.score.local_score(Y, old_parents)
            new_score = self.score.local_score(Y, new_parents)
            score_diff = new_score - old_score
            yield operation, score_diff

        # all potential operation for reverse edge
        for (X, Y) in dag.edges:
            operation = ('rev', (X, Y))
            try:
                temp_dag = dag.copy()
                temp_dag.remove_edge(X, Y)
                temp_dag.add_edge(Y, X)
                cycles = list(nx.find_cycle(temp_dag))
            except nx.NetworkXNoCycle:
                cycles = []
            if cycles:
                continue
            else:
                old_x_parents = list(dag.predecessors(X))
                new_x_parents = old_x_parents + [Y]
                old_y_parents = list(dag.predecessors(Y))
                new_y_parents = old_y_parents
                new_y_parents.remove(X)
                new_score = self.score.local_score(Y, new_y_parents) + self.score.local_score(X, new_x_parents)
                old_score = self.score.local_score(Y, old_y_parents) + self.score.local_score(X, old_x_parents)
                score_diff = new_score - old_score
                yield operation, score_diff

    def run(self):
        start_dag = self.initial_DAG
        if self.show_process:
            iteration = trange(self.max_iter)
        else:
            iteration = range(int(self.max_iter))
        current_dag = start_dag
        for _ in iteration:
            best_operation, best_score_diff = max(self.legal_operation(current_dag), key=lambda x: x[1])
            if best_operation[0] == '+':
                current_dag.add_edge(*best_operation[1])
            elif best_operation[0] == '-':
                current_dag.remove_edge(*best_operation[1])
            elif best_operation[0] == 'rev':
                X, Y = best_operation[1]
                current_dag.remove_edge(X, Y)
                current_dag.add_edge(Y, X)

        return current_dag


if __name__ == '__main__':
    pddata = pd.read_csv(r"../datasets/Asian.csv",index_col=False)
    hc = HillClimb(BicScore, pddata,max_iter=500)
    result = hc.run()
    print(result.edges)

