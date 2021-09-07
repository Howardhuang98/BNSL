#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   order_graph.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/29 14:03  
------------      
"""
from itertools import permutations

from dlbn.base.score import *


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

    def add_cost(self, score_method: Score, data: pd.DataFrame):
        if not self.edges:
            raise ValueError("please run generate_order_graph")
        for _, edge in enumerate(self.edges):
            print("{}个边".format(_))
            u = edge[0]
            v = edge[1]
            # new added node: x
            x = str(list(v - u)[0])
            # get optimal parents out of u
            if u:
                pg = ParentGraph(x, list(u))
                pg.generate_order_graph()
                pg.add_cost(MDL_score, data)
                optimal_parents, cost = pg.find_optimal_parents()
                self.add_edge(u, v, cost=cost, optimal_parents=optimal_parents)
            else:
                self.add_edge(u, v, cost=0, optimal_parents=frozenset())

        return self

    def find_shortest_path(self):
        pass


class ParentGraph(OrderGraph):

    def __init__(self, variable: str, potential_parents: list):
        super(ParentGraph, self).__init__(potential_parents)
        self.potential_parents = potential_parents
        self.variable = variable

    def add_cost(self, score_method: Score, data: pd.DataFrame):
        """
        edge 的存储形式：(frozenset(), frozenset({'bronc'}), {'cost': 8.517193191416238})
        :param score_method:
        :param data:
        :return:
        """
        score = score_method(data)
        self.generate_order_graph()
        for edge in self.edges:
            parents = list(edge[1])
            cost = score.local_score(self.variable, parents)
            u = edge[0]
            v = edge[1]
            self.add_edge(u, v, cost=cost)
        return self

    def find_optimal_parents(self):
        if not self.edges:
            raise ValueError("Parents graph is empty, please run add_cost() !")
        else:
            optimal_tuple = max(self.edges.data(), key=lambda x: x[2]["cost"])
            optimal_parents = optimal_tuple[1]
            cost = optimal_tuple[2]['cost']
        return optimal_parents, cost


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    variables = list(data.columns)
    og = OrderGraph(variables)
    og.generate_order_graph()
    og.add_cost(MDL_score, data)
    print(og.edges.data())
