#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   order2graph.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/17 19:36  
------------      
"""
import itertools
from typing import List

from BNSL.bnsl.base import Score
from BNSL.bnsl.graph import DAG


def order2graph(order: List, score_method: Score):
    g = DAG()
    # add nodes
    g.add_nodes_from(order)
    # add parents:
    for i, node in enumerate(order):
        curr = 0
        curr_par = []
        for num_par in range(i):
            for par in itertools.combinations(order[:i], num_par):
                score = score_method.local_score(node, tuple(par))
                if score > curr:
                    curr = score
                    curr_par = par
        for p in curr_par:
            g.add_edge(p, node)
    return g


if __name__ == '__main__':
    order2graph()
