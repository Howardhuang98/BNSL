#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   order_graph.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/29 14:03  
------------      
"""
from itertools import permutations

from dlbn.estimator.score_function import *


class OrderGraph(DAG):
    def __init__(self, variables: list):
        self.variables = variables
        super(OrderGraph, self).__init__()

    def generate_order_graph(self):
        for order in permutations(self.variables):
            previous = []
            previous_name = frozenset(previous)
            self.add_node(previous_name)
            for node in order:
                if previous == []:
                    node_name = frozenset([node])
                    self.add_node(node_name)
                    self.add_edge(previous_name, node_name)
                    previous = [node]
                    previous_name = frozenset(previous)
                else:
                    node_name = frozenset(previous + [node])
                    self.add_node(node_name)
                    self.add_edge(previous_name, node_name)
                    previous = previous + [node]
                    previous_name = frozenset(previous)
        return self

    def add_cost(self, score_function: Score, contingency_table: pd.DataFrame):
        score = score_function(contingency_table)
        for u, v in self.edges:
            if not u:
                self.add_edge(u, v, cost=0, optimal_parents=[])
                continue
            child = str(list(v - u)[0])
            list_u = list(u)
            parents, cost = score.find_optimal_parents(child, list_u)
            self.add_edge(u, v, cost=cost, optimal_parents=parents)


if __name__ == '__main__':
    data = pd.read_excel(r"../datasets/simple.xlsx")
    data = Data(data)
    contb = data.contingency_table()
    og = OrderGraph(data.variables)
    og.generate_order_graph()
    og.add_cost(BicScore, contb)
    for info in og.edges.data():
        print("{}-->{}, cost = {}, optimal parents are {}".format(info[0], info[1], info[2]['cost'],
              info[2]['optimal_parents']))
