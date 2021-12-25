#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pc.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/25 21:25  
------------      
"""
from itertools import combinations, permutations

import pandas as pd

from cit import fisherz
import networkx as nx


def create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def estimate_skeleton(data:pd.DataFrame, alpha=0.05 ,indep_test_func=fisherz, **kwargs):
    """Estimate a skeleton graph
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(len(data.columns))
    node_size = len(data.columns)
    sep_set = [[set() for i in node_ids] for j in range(node_size)]
    g = create_complete_graph(node_ids)
    data_matrix = data.values

    fixed_edges = set()

    l = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            if (i, j) in fixed_edges:
                continue

            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    p_val = indep_test_func(data_matrix, i, j, set(k),
                                            **kwargs)
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return g, sep_set

if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    g,sep = estimate_skeleton(data)
    print(g.edges)
    print(sep)