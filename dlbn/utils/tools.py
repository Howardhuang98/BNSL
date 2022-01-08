#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tools.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/30 22:39  
------------      
"""
from dlbn.graph import DAG


def edge2adj(edge_path, nodes, adj_path):
    """
    convert edge csv file into adj xlsx file
    :param edge_path:
    :param nodes:
    :param adj_path:
    :return:
    """
    g = DAG()
    g.read(edge_path)
    g.add_nodes_from(nodes)
    g.to_excel_DataFrame(adj_path)
