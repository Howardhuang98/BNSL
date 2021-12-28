#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/25 14:50
------------
"""
import math
from itertools import permutations
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def acc(dag, true_dag):
    """

    :param dag0:
    :param dag1:
    :return:
    """
    FP = len(set(dag.edges) - set(true_dag.edges))
    FN = len(set(true_dag.edges) - set(dag.edges))
    TP = len(set(true_dag.edges) & set(dag.edges))
    TN = (len(dag.nodes) ** 2 - len(dag.nodes)) - FP - FN - TP
    return (TP + TN) / (FP + FN + TP + TN)


class DAG(nx.DiGraph):
    """
    inherit class nx.DiGraph
    """

    def __init__(self, incoming_graph_data=None, **kwargs):
        super(DAG, self).__init__(incoming_graph_data, **kwargs)
        cycle = self._check_cycle()
        if cycle:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycle])
            raise ValueError(out_str)

    def _check_cycle(self):
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            return False
        else:
            return cycles

    def score(self, score_method, data: pd.DataFrame, detail=False):
        score_dict = {}
        score_list = []
        if len(self.nodes) < len(data.columns):
            print("Isolated node detected, this function will add nodes")
            self.add_nodes_from(data.columns)
        for node in self.nodes:
            parents = list(self.predecessors(node))
            s = score_method(data)
            local_score = s.local_score(node, parents)
            score_list.append(local_score)
            if detail:
                score_dict[node] = local_score
        if detail:
            return sum(score_list), score_dict
        return sum(score_list)

    def to_excel(self, path: str):
        edge_list = self.edges
        edges_data = pd.DataFrame(columns=['source node', 'target node'])
        for edge_pair in edge_list:
            edges_data.loc[edges_data.shape[0]] = {'source node': edge_pair[0], 'target node': edge_pair[1]}
        edges_data.to_excel(path)
        return None

    def __sub__(self, other):
        """
        Use structure Hamming Distance (SHD) to subtract.
        SHD = FP + FN
        FP: The number of edges discovered in the learned graph that do not exist in the true graph(other)
        FN: The number of direct independence discovered in the learned graph that do not exist in the true graph.
        :param other:
        :return:
        """
        if isinstance(self, type(other)):
            FP = len(set(self.edges) - set(other.edges))
            FN = len(set(other.edges) - set(self.edges))
            return FP + FN

        else:
            raise ValueError("cannot subtract DAG instance with other instance")

    def read(self, path: str, source='source node', target='target node'):
        """
        here we need excel written in this format:

            source node   target node
        1       a            b
        2       a            c
       ...       ...          ...
        :param path:
        :return:
        """
        if path.endswith("xlsx"):
            data = pd.read_excel(path)
        elif path.endswith("csv"):
            data = pd.read_csv(path)
        else:
            raise ValueError()
        edge_list = []
        for row_tuple in data.iterrows():
            u = row_tuple[1][source]
            v = row_tuple[1][target]
            edge_list.append((u, v))
        self.add_edges_from(edge_list)
        return self

    def show(self, score_method, data: pd.DataFrame):
        nx.draw_networkx(self)
        plt.title("Bayesian network with Score={}".format(self.score(score_method, data)))
        plt.show()
        return None

    def legal_operations(self):
        """
        iterator
        yield all legal operations
        :return:
        """

        potential_new_edges = (set(permutations(list(self.nodes), 2)) - set(self.edges()) - set(
            [(v, u) for (u, v) in self.edges()]))

        for u, v in potential_new_edges:
            if not nx.has_path(self, v, u):
                operation = ('+', (u, v))
                yield operation

        for u, v in self.edges:
            operation = ('-', (u, v))
            yield operation

        for u, v in self.edges:
            if not any(map(lambda path: len(path) > 2, nx.all_simple_paths(self, u, v))):
                operation = ('flip', (u, v))
                yield operation

    def score_delta(self, operation, score_method):
        opera, uv = operation[0], operation[1]
        u, v = uv[0], uv[1]
        s = score_method
        if opera == '+':
            old_parents = list(self.predecessors(v))
            new_parents = old_parents + [u]
            score_delta = s.local_score(v, new_parents) - s.local_score(v, old_parents)
            return score_delta
        if opera == '-':
            old_parents = list(self.predecessors(v))
            new_parents = old_parents[:]
            new_parents.remove(u)
            score_delta = s.local_score(v, new_parents) - s.local_score(v, old_parents)
            return score_delta
        if opera == 'flip':
            old_v_parents = list(self.predecessors(v))
            old_u_parents = list(self.predecessors(u))
            new_u_parents = old_u_parents + [v]
            new_v_parents = old_v_parents[:]
            new_v_parents.remove(u)
            score_delta = (s.local_score(v, new_v_parents) + s.local_score(u, new_u_parents) - s.local_score(v,
                                                                                                             old_v_parents) - s.local_score(
                u, old_u_parents))
            return score_delta

    def do_operation(self, operation):
        if operation[0] == '+':
            self.add_edge(*operation[1])
        if operation[0] == '-':
            self.remove_edge(*operation[1])
        if operation[0] == 'flip':
            u, v = operation[1]
            self.remove_edge(u, v)
            self.add_edge(v, u)
        return None

    @property
    def adj_matrix(self):
        return nx.to_numpy_array(self, dtype=int)

    @property
    def genome(self):
        return self.adj_matrix.flatten()

    def from_genome(self, genome, node_labels):
        num_node = int(math.sqrt(len(genome)))
        adj_matrix = np.reshape(genome, newshape=(num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if i == j:
                    continue
                elif adj_matrix[i, j] == 1:
                    self.add_edge(node_labels[i], node_labels[j])
                elif adj_matrix[i, j] == 0:
                    try:
                        self.remove_edge(node_labels[i], node_labels[j])
                    except:
                        continue
        return self


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
        generate order graph. if there is num_of_nodes variable, there will be 2^num_of_nodes-1 states(nodes) in graph
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

    def add_cost(self, score_method, data: pd.DataFrame, num_of_workers=4, **kwargs):
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
                print(u, v, cost)
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

    def add_cost(self, score_method, data: pd.DataFrame):
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
    g = DAG()
    g.from_genome([1, 1, 1, 0],['a','b'])
    print(g.genome)
