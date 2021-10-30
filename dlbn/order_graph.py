#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   order_graph.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/29 14:03  
------------      
"""
import logging
from itertools import permutations
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from dlbn.score import *
from dlbn.direct_graph import *

"""
OrderGraph class
ParentGraph class

            workflow:
            
            OrderGraph
                |
            generate order graph
                |                       |-parent graph
            add cost on order graph ----|-add cost on parent graph
                |                       |-find optimal parents
            find shortest path


"""


class OrderGraph(DAG):
    """
    Order graph class
    base on a list of variable, initialize an order graph.

    """

    def __init__(self, variables: list):
        self.variables = variables
        self.shortest_path = None
        super(OrderGraph, self).__init__()

    def generate_order_graph(self):
        """
        generate order graph. if there is n variable, there will be 2^n-1 states(nodes) in graph
        """
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

    @classmethod
    def _cost_on_u_v(cls, dic):
        """
        required by add_cost function. Because the map() only can pass one argument into function
        u, v, score_method, data: pd.DataFrame
        u is a frozenset()
        v is a frozenset()

        return dict
        """
        # get all argument we need
        u = dic['u']
        v = dic['v']
        score_method = dic['score_method']
        data = dic['data']
        # store result as networkx format:
        res = {'u_of_edge': u, 'v_of_edge': v}
        # new added node: x
        x = str(list(v - u)[0])
        # get optimal parents out of u

        pg = ParentGraph(x, list(u))
        pg.generate_order_graph()
        pg.add_cost(score_method, data)
        optimal_parents, cost = pg.find_optimal_parents()
        res['cost'] = cost
        res['optimal_parents'] = optimal_parents
        return res

    def add_cost(self, score_method: Score, data: pd.DataFrame, num_of_workers=4, **kwargs):
        """
        use score method to add cost on edges.
        :param score_method:
        :param data:
        :num_of_worker
        :return:
        """
        if not self.edges:
            raise ValueError("please run generate_order_graph")
        with Pool(processes=num_of_workers) as pool:
            arg_list = []
            for edge in self.edges:
                arg = {'u': edge[0], 'v': edge[1], 'score_method': score_method, 'data': data}
                arg_list.append(arg)
            result = pool.map(self._cost_on_u_v, tqdm(arg_list, desc="Adding cost"))
            for res in result:
                self.add_edge(**res)
        return self

        # 串行方式进行add cost
        # for edge in tqdm(self.edges, desc="Adding cost", colour='green', miniters=1):
        #     u = edge[0]
        #     v = edge[1]
        #     # new added node: x
        #     x = str(list(v - u)[0])
        #     # get optimal parents out of u
        #     if u:
        #         pg = ParentGraph(x, list(u))
        #         pg.generate_order_graph()
        #         pg.add_cost(score_method, data)
        #         optimal_parents, cost = pg.find_optimal_parents()
        #         self.add_edge(u, v, cost=cost, optimal_parents=optimal_parents)
        #         logging.info("{}->{},cost={},optimal_parents={}".format(u, v, cost, optimal_parents))
        #     else:
        #         self.add_edge(u ,v, cost=0, optimal_parents=frozenset())
        #
        # return self

    def find_shortest_path(self):
        start = frozenset()
        end = frozenset(self.variables)
        shortest_path = nx.dijkstra_path(self, start, end, weight='cost')
        self.shortest_path = shortest_path
        return shortest_path

    def optimal_result(self):
        """
        store the optimal result
        :return:
        """
        if not self.shortest_path:
            raise ValueError("please run find_shortest_path()")
        else:
            result_dag = DAG()
            cost_list = []

            for i in range(len(self.shortest_path) - 1):
                u = self.shortest_path[i]
                v = self.shortest_path[i + 1]
                cost = self.edges[u, v]['cost']
                optimal_parents = list(self.edges[u, v]['optimal_parents'])
                variable = str(list(v - u)[0])
                if optimal_parents:
                    for parent in optimal_parents:
                        result_dag.add_edge(parent, variable)
                else:
                    result_dag.add_node(variable)
                cost_list.append(cost)
        return result_dag


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
        for node in self.nodes:
            parents = list(node)
            cost = score.local_score(self.variable, parents)
            self.add_node(node, cost=cost)
        return self

    def find_optimal_parents(self):
        optimal_tuple = min(self.nodes.data(), key=lambda x: x[1]["cost"])
        optimal_parents = optimal_tuple[0]
        cost = optimal_tuple[1]['cost']
        return optimal_parents, cost


if __name__ == '__main__':
    data = pd.read_excel('../datasets/simple.xlsx')
    variables = list(data.columns)
    og = OrderGraph(variables)
    og.generate_order_graph()
    og.add_cost(MDL_score, data)
    og.find_shortest_path()
    print(og.edges.data())
