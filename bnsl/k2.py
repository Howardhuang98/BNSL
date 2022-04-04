#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   k2.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/29 20:24  
------------      
"""
import pandas as pd
from bnsl.graph import DAG
from bnsl.base import Score


def order_to_dag(order, u: int, score_method: Score):
    dag = DAG()
    for i in range(len(order)):
        node = order[i]
        if i == 0:
            dag.add_node(node)
        else:
            pre = order[:i]
            flag = True
            parents = []
            while flag and len(parents) <= u:
                try:
                    parents = find_z(node,parents,pre,score_method)
                except ValueError:
                    flag = False
            if parents:
                for parent in parents:
                    dag.add_edge(parent, node)
    return dag



def find_z(node:str,parents:list,pre:list,score_method:Score):
    possible_set = set(pre)-set(parents)
    score = score_method.local_score(node, parents)
    score_updated = False
    for z in possible_set:
        new_parents = parents + [z]
        s = score_method.local_score(node, new_parents)
        if s >= score:
            score = s
            parents = new_parents
            score_updated = True
    if score_updated:
        return parents
    else:
        raise ValueError






