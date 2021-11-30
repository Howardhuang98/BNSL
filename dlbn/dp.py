#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dp.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/30 13:20  
------------      
"""
import itertools

import pandas as pd
from score import *


def generate_parent_graph(data:pd.DataFrame,score_method:Score):
    """

    :param data:
    :return:
    """
    s = score_method(data)
    parent_graph = {}
    x_parent_graph = {}
    for x in data.columns:
        parent_set = set(data.columns) - set([x])
        for j in range(len(set(parent_set))+1):
            if j == 0:
                best_score = s.local_score(str(x),[])
                x_parent_graph[tuple([])] = best_score
            else:
                for node in itertools.combinations(parent_set, j):
                    best_score = s.local_score(str(x),list(node))
                    x_parent_graph[node] = best_score
        parent_graph[x] = x_parent_graph

    return parent_graph

def query_best_score(x_parent_graph,node,score):
    for previous_node in x_parent_graph.keys():
        if set(previous_node).issubset(set(node)):
            if x_parent_graph[previous_node] > score:
                return x_parent_graph[previous_node]




if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    pg = generate_parent_graph(data,BIC_score)
    print(pg['asia'])