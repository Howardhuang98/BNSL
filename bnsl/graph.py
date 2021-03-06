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

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import permutation
from tqdm import tqdm

from bnsl.base import Score


def acc(dag, true_dag):
    """

    :param true_dag:
    :param dag:
    :return:
    """
    FP = len(set(dag.edges) - set(true_dag.edges))
    FN = len(set(true_dag.edges) - set(dag.edges))
    TP = len(set(true_dag.edges) & set(dag.edges))
    TN = (len(dag.nodes) ** 2 - len(dag.nodes)) - FP - FN - TP
    return (TP + TN) / (FP + FN + TP + TN)


class DAG(nx.DiGraph):
    """
    The graph class used in this project, inherited class nx.DiGraph.
    When it is initialized, it has to be acyclic.
    """

    def __init__(self, incoming_graph_data=None, **kwargs):
        """

        :param incoming_graph_data: input graph (optional, default: None). Data to initialize graph. If None (
        default) an empty graph is created. The data can be any format that is supported by the to_networkx_graph()
        function, currently including edge list, dict of dicts, dict of lists, NetworkX graph, NumPy matrix or 2d
        ndarray, SciPy sparse matrix, or PyGraphviz graph.
        :param kwargs:

        for example:
        dag = DAG()
        """
        super(DAG, self).__init__(incoming_graph_data, **kwargs)
        cycle = self._check_cycle()
        if cycle:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycle])
            raise ValueError(out_str)
        self.calculated_score = None

    def _check_cycle(self):
        """
        Also known as "has cycle(s)"
        :return: False if there is no cycle
                 node list if there is cycle
        """
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            return False
        else:
            return cycles


    def score(self, score_method, detail=False):
        """

        :param score_method: score criteria
        :param data: observed data
        :param detail: whether return a dictionary containing every local score. Default is False
        :return: score or (score, dict)
        """
        score_dict = {}
        score_list = []
        if not isinstance(score_method, Score):
            raise ValueError("Input the score method instance")
        if len(self.nodes) < len(score_method.data.columns):
            self.add_nodes_from(score_method.data.columns)
        for node in self.nodes:
            parents = list(self.predecessors(node))
            score = score_method.local_score(node, parents)
            score_list.append(score)
            if detail:
                score_dict[node] = score
        self.calculated_score = sum(score_list)
        if detail:
            return self.calculated_score, score_dict
        return self.calculated_score

    def to_excel(self, path: str, source='source node', target='target node'):
        """
        Save the edges of DAG in to excel file, notice that this file may lose node(s).

        :param path: the io path to save
        :param source: column name of source node, default as source node
        :param target: column name of target node, default as target node
        :return: None
        """
        edge_list = self.edges
        edges_data = pd.DataFrame(columns=[source, target])
        for edge_pair in edge_list:
            edges_data.loc[edges_data.shape[0]] = {source: edge_pair[0], target: edge_pair[1]}
        edges_data.to_excel(path)
        return None

    def to_csv(self, path: str, source='source node', target='target node'):
        edge_list = self.edges
        edges_data = pd.DataFrame(columns=[source, target])
        for edge_pair in edge_list:
            edges_data.loc[edges_data.shape[0]] = {source: edge_pair[0], target: edge_pair[1]}
        edges_data.to_csv(path)
        return None

    def to_csv_adj(self, path: str):
        df = nx.to_pandas_adjacency(self)
        df.to_csv(path)
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
        Notice: file only contain edges, the DAG instance may lose isolated nodes.
        here we need excel written in this format:

            source node   target node
        1       a            b
        2       a            c
       ...       ...          ...
        :param path: io path
        :param source: column name of source node
        :param target: column name of target node
        :return: DAG instance
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

    def read_DataFrame_adjacency(self, path: str):
        if path.endswith("xlsx"):
            df = pd.read_excel(path, index_col=0)
        elif path.endswith("csv"):
            df = pd.read_csv(path, index_col=0)
        self.add_nodes_from(df.columns)
        edges = ((df.columns[int(e[0])], df.columns[int(e[1])]) for e in zip(*df.values.nonzero()))
        self.add_edges_from(edges)
        return self

    def show(self):
        """
        draw the figure of DAG
        :return: None
        """
        nx.draw_networkx(self)
        plt.show()
        return None

    def summary(self):
        """
        return a string to describe current dag.
        :return: a description string
        """
        print(
            """
            DAG summary:
            Number of nodes: {},
            Number of edges: {},
            Adjacency matrix: {},
            
            """.format(len(self.nodes), len(self.edges), self.adj_matrix)
        )

    def legal_operations(self):
        """
        iterator, yield all legal operations like ('+', (u, v)), ('-', (u, v)), ('flip', (u, v)).
        :return: iterable, operation ('+', (u, v))
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

    def adj_DataFrame(self, **kwargs):
        return nx.to_pandas_adjacency(self, **kwargs)

    @property
    def adj_df(self, **kwargs):
        return nx.to_pandas_adjacency(self, **kwargs)

    @property
    def genome(self):
        return self.adj_matrix.flatten()

    def random_dag(self, nodes=None, seed=None, num_parents=None):
        if not num_parents:
            num_parents = len(self.nodes)
        if seed:
            np.random.seed(seed)
        if nodes is not None:
            nodes = permutation(nodes)
            num_parents = len(nodes)
        else:
            self.remove_edges_from([i for i in self.edges])
            nodes = permutation(list(self.nodes))
        for i in range(len(nodes)):
            v = nodes[i]
            num_parents = np.random.randint(0, num_parents + 1)
            parent_list = permutation(nodes[i + 1:])[:num_parents]
            for pa in parent_list:
                self.add_edge(pa, v)
            if parent_list.size == 0:
                self.add_node(v)
        return self


if __name__ == '__main__':
    g = DAG()
    g.from_genome([1, 1, 1, 0, 1, 0], ['a', 'b', 'c', 'd'])
    print(g.genome)
