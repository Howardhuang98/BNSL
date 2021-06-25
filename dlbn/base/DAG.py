#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/25 14:50
------------
"""
import networkx as nx
import pandas as pd


class DAG(nx.DiGraph):
    """
    inherit class nx.DiGraph
    """

    def __init__(self, incoming_graph_data=None):
        super(DAG, self).__init__(incoming_graph_data)
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

    def to_excel(self, path: str):
        edge_list = self.edges
        edges_data = pd.DataFrame(columns=['source node', 'target node'])
        for edge_pair in edge_list:
            edges_data.loc[edges_data.shape[0]] = {'source node': edge_pair[0], 'target node': edge_pair[1]}
        edges_data.to_excel(path)
        return None
