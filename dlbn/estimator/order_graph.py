#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   order_graph.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/29 14:03  
------------      
"""
from itertools import permutations

from dlbn import DAG


class OrderGraph(DAG):
    def __init__(self, variables: list):
        self.variables = variables
        super(OrderGraph, self).__init__()

    def generate_order_graph(self):
        for order in permutations(self.variables):
            previous = ['null']
            previous_name = frozenset(previous)
            self.add_node(previous_name)
            for node in order:
                if previous == ['null']:
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


if __name__ == '__main__':
    variables = ['X1', 'X2', 'X3','X4']
    a = OrderGraph(variables)
    a.generate_order_graph()
    print(len(a.nodes))
    print(a.edges)
