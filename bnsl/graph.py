#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/25 14:50
------------
"""
import random
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .base import Score


def compare(dag, true_dag):
    """
    Compare two dag.

    Args:
        dag: DAG instance, usually predicted DAG.
        true_dag: DAG instance, the ground truth DAG.

    Returns:
        dict: a dict containing accuracy, precision, recall, SHD, norm
    """
    FP = len(set(dag.edges) - set(true_dag.edges))
    FN = len(set(true_dag.edges) - set(dag.edges))
    TP = len(set(true_dag.edges) & set(dag.edges))
    # The TN might be wrong.
    TN = (len(dag.nodes) ** 2 - len(dag.nodes)) - FP - FN - TP
    accuracy = (TP + TN) / (FP + FN + TP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = recall * precision * 2 / (precision + recall)
    node_list = true_dag.nodes
    SHD = np.sum(np.absolute(dag.adj_np(node_list=node_list) - true_dag.adj_np(node_list=node_list)))
    norm = np.linalg.norm(dag.adj_np(node_list=node_list) - true_dag.adj_np(node_list=node_list))
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': F1, 'SHD': SHD, 'norm': norm}


def random_dag(node_list, num_parents=None):
    """
    Randomly generate DAG instance.

    Args:
        node_list: node list.
        num_parents: number of parents

    Returns:
        A dag
    """
    dag = DAG()
    random.shuffle(node_list)
    for i in range(1, len(node_list)):
        if not num_parents:
            num_par = np.random.randint(0, i + 1)
        else:
            num_par = min(np.random.randint(0, num_parents), i)
        par = random.sample(node_list[:i], num_par)
        if not par:
            dag.add_node(node_list[i])
        else:
            for p in par:
                dag.add_edge(p, node_list[i])
    return dag


class DAG(nx.DiGraph):
    """
    The graph class used in this project, inherited class nx.DiGraph.
    When it is initialized, it has to be acyclic.
    """

    def __init__(self, incoming_graph_data=None, **kwargs):
        """
        Initialize a DAG instance.

        for example:
        dag = DAG()

        Args:
            incoming_graph_data: input graph (optional, default: None). Data to initialize graph. If None (
            default) an empty graph is created. The data can be any format that is supported by the to_networkx_graph()
            function, currently including edge list, dict of dicts, dict of lists, NetworkX graph, NumPy matrix or 2d
            ndarray, SciPy sparse matrix, or PyGraphviz graph.
            **kwargs:
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
        Check cycles.

        Returns:
            False if there is no cycle, node list if there is cycle
        """
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            return False
        else:
            return cycles

    def score(self, score_method, detail=False):
        """
        Calculate the score of the DAG.

        Args:

            score_method: score criteria instance
            detail: return a dictionary containing every local score. Default is False.

        Returns:
            score of the DAG (or score, dict)
        """
        score_dict = {}
        score_list = []
        if not isinstance(score_method, Score):
            raise ValueError("Input the score method instance")
        if len(self.nodes) < len(score_method.data.columns):
            self.add_nodes_from(score_method.data.columns)
        for node in self.nodes:
            parents = tuple(self.predecessors(node))
            score = score_method.local_score(node, parents)
            score_list.append(score)
            if detail:
                score_dict[node] = score
        self.calculated_score = sum(score_list)
        if detail:
            return self.calculated_score, score_dict
        return self.calculated_score

    def __sub__(self, other):
        """
        Use structure Hamming Distance (SHD) to subtract.
        SHD = FP + FN
        FP: The number of edges discovered in the learned graph that do not exist in the true graph(other)
        FN: The number of direct independence discovered in the learned graph that do not exist in the true graph.

        Args:

            other: DAG

        Returns:

            SHD value
        """
        if isinstance(self, type(other)):
            FP = len(set(self.edges) - set(other.edges))
            FN = len(set(other.edges) - set(self.edges))
            return FP + FN

        else:
            raise ValueError("cannot subtract DAG instance with other instance")

    def read(self, path: str, mode='edge_list', source='source node', target='target node'):
        """
        Read DAG from csv file.There are two modes: edge_list, adjacent_matrix.

        In the `edge_list` mode, the csv file looks like:
        ::

            source node,target node
            asia,tub
            tub,either
            either,dysp
            either,xray

        In the `adjacent_matrix` mode, the csv file looks like:
        ::

            ,asia,tub,either,smoke,lung,bronc,dysp,xray
            asia,0,1,0,0,0,0,0,0
            tub,0,0,1,0,0,0,0,0
            either,0,0,0,0,0,0,1,1
            smoke,0,0,0,0,1,1,0,0
            lung,0,0,1,0,0,0,0,0
            bronc,0,0,0,0,0,0,1,0
            dysp,0,0,0,0,0,0,0,0
            xray,0,0,0,0,0,0,0,0

        Args:
            path: file path.
            mode: `edge_list` mode or `adjacent_matrix` mode.
            source: the source node column.
            target: the target node column.
        """
        if mode == 'edge_list':
            data = pd.read_csv(path)
            edge_list = []
            for row_tuple in data.iterrows():
                u = row_tuple[1][source]
                v = row_tuple[1][target]
                edge_list.append((u, v))
            self.add_edges_from(edge_list)
        elif mode == 'adjacent_matrix':
            data = pd.read_csv(path, index_col=0)
            self.add_nodes_from(data.columns)
            edges = ((data.columns[int(e[0])], data.columns[int(e[1])]) for e in zip(*data.values.nonzero()))
            self.add_edges_from(edges)
        else:
            raise ValueError("Mode error!")

    def save(self, path: str, mode='edge_list'):
        """
        Save the DAG into csv file.

        Args:
            path: file path.
            mode: `edge_list` mode or `adjacent_matrix` mode.
        """
        if mode == 'edge_list':
            with open(path, "w") as f:
                f.write("source node,target node\n")
                for u, v in self.edges:
                    f.write(str(u) + "," + str(v) + "\n")
        elif mode == 'adjacent_matrix':
            self.adj_df().to_csv(path)

    def show(self):
        """
        Draw the figure of DAG

        """
        nx.draw_networkx(self)
        plt.show()
        return None

    def summary(self):
        """
        Print a summary to describe current dag.
        """
        print(f"DAG summary:\nNumber of nodes: {len(self.nodes)}\nNumber of edges: {len(self.edges)}\n")

    def legal_operations(self):
        """
        Iterator, yield all legal operations like ('+', (u, v)), ('-', (u, v)), ('flip', (u, v)).

        Yields:
            operation ('+', (u, v))
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

    def adj_np(self, node_list=None):
        """
        The adjacent matrix in the form of np.array of DAG.

        Returns:
            np.array
        """
        return nx.to_numpy_array(self, dtype=int, nodelist=node_list)

    def adj_df(self, node_list=None):
        """
        The adjacent matrix in the form of pd.Dataframe of DAG.
        Args:
            node_list:

        Returns:
            pd.Dataframe
        """
        return nx.to_pandas_adjacency(self, dtype=int, nodelist=node_list)

    def genome(self):
        """
        The genome of DAG

        Returns:
            np.array
        """
        return self.adj_np.flatten()
