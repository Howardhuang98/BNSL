#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dp.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/30 13:20  
------------      
"""
import itertools

import networkx as nx

from dlbn.graph import DAG
from dlbn.score import *


def sort_tuple(t: tuple):
    l = list(t)
    l.sort()
    return tuple(l)


def generate_parent_graph(data: pd.DataFrame, score_method: Score):
    """

    :param score_method:
    :param data:
    :return:
    """
    s = score_method(data)
    parent_graph = {}
    x_parent_graph = {}
    for x in data.columns:
        parent_set = set(data.columns) - {x}
        # j layers from 0 to len(set(parent_set))
        for j in range(len(set(parent_set)) + 1):
            if j == 0:
                best_score = calculate_best_score(str(x), [], x_parent_graph, s)
                x_parent_graph[tuple()] = best_score
            else:
                for node in itertools.combinations(parent_set, j):
                    node = sort_tuple(node)
                    best_score = calculate_best_score(str(x), list(node), x_parent_graph, s)
                    x_parent_graph[node] = best_score
        parent_graph[x] = x_parent_graph
        x_parent_graph = {}

    return parent_graph


def calculate_best_score(x, parents, x_parent_graph, score_instance):
    best_score = score_instance.local_score(str(x), list(parents))
    if len(parents) > 0:
        for ancestor in itertools.combinations(parents, len(parents) - 1):
            ancestor = sort_tuple(ancestor)
            if x_parent_graph[ancestor] < best_score:
                best_score = x_parent_graph[ancestor]
    return best_score


def generate_order_graph(data, parent_graph):
    order_graph = nx.DiGraph()
    for i in range(len(data.columns) + 1):
        for subset in itertools.combinations(set(data.columns), i):
            subset = sort_tuple(subset)
            order_graph.add_node(subset)
            for variable in subset:
                parents = tuple(v for v in subset if (v != variable))
                parents = sort_tuple(parents)
                best_score = parent_graph[variable][parents]
                structure = query_best_structure(variable, parents, parent_graph)
                order_graph.add_edge(parents, subset, weight=best_score, structure=structure)
    return order_graph


def query_best_structure(variable, parents, parent_graph):
    x_parent_graph = parent_graph[variable]
    best_score = parent_graph[variable][parents]
    structure = parents
    for i in range(len(parents) - 1, -1, -1):
        for ancestor in itertools.combinations(parents, i):
            if x_parent_graph[tuple(ancestor)] == best_score:
                structure = ancestor
    return structure


def order2dag(order_graph, data):
    result_dag = DAG()
    path = nx.shortest_path(order_graph, weight='weight', source=tuple(), target=sort_tuple(tuple(data.columns)))
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        print(u, v, order_graph.edges[u, v]['weight'], order_graph.edges[u, v]['structure'])
        parents = list(order_graph.edges[u, v]['structure'])
        variable = tuple(set(v) - set(u))[0]
        if parents:
            for node in parents:
                result_dag.add_edge(node, variable)
        else:
            result_dag.add_node(variable)
    return result_dag


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    pg = generate_parent_graph(data, MDL_score)
    og = generate_order_graph(data, pg)
    dag = order2dag(og, data)
    print(dag.edges)
    print(dag.score(MDL_score, data))
    nx.draw_networkx(dag)
    plt.show()
