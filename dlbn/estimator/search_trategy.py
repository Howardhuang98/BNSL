#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   search_trategy.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/28 15:30  
------------      
"""
from itertools import permutations

from score_function import *


class SearchStrategy:
    def __init__(self):
        pass

    def run(self):
        pass


class HillClimb(SearchStrategy):
    def __init__(self, score_function, data):
        """
        hill climb search
        :param score_function: BicScore,MdlScore
        :param data: pd.Dataframe or Data
        """
        super(HillClimb, self).__init__()
        if isinstance(data, pd.DataFrame):
            data = Data(data)
        self.data = data
        self.score = score_function(data.contingency_table())

    def legal_operation(self, dag: DAG):
        """
        a generator that generates a list of legal operation with its score difference
        :param dag:
        :param DAG
        :yield (legal operation, score diff)
        """
        dag.add_nodes_from(self.data.variables)
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


if __name__ == '__main__':
    pddata = pd.read_excel(r"../test/test_data.xlsx")
    hc = HillClimb(BicScore, pddata)
    g = DAG()
    g.add_edges_from([('A', 'B'), ('C', 'B')])
    generator = hc.legal_operation(g)
    print(type(generator))
    print(max(generator))
