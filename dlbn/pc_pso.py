#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pc_pso.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/27 11:46  
------------      
"""
import math

import networkx
import numpy as np

from dlbn.graph import DAG
from dlbn.pc import *


def adj_to_genome(adj_matrix):
    l = adj_matrix.shape[0] ** 2
    genome = np.reshape(adj_matrix, l)
    return genome


def genome_to_adj(genome):
    num_node = int(math.sqrt(len(genome)))
    adj = genome.reshape(shape=(num_node, num_node))
    return adj


def cal_bic_of_gen(genome, nodes):
    adj = genome_to_adj(genome)
    DAG(adj, nodes)


def mutation(genome, rate):
    for index, a in enumerate(genome):
        r = np.random.uniform()
        if r < rate:
            if a == 1:
                genome[index] = 0
            else:
                genome[index] = 1
    return genome


def crossover(a_genome, b_genome):
    child = []
    for i in range(len(a_genome)):
        r = np.random.uniform()
        if r > 0.5:
            child.append(a_genome[i])
        else:
            child.append(b_genome[i])
    child = np.asarray(child)
    return child


def pcpso(data: pd.DataFrame, npop=100, maxit=10000, **kwargs):
    """

    :param data:
    :param npop:
    :param maxit:
    :return:
    """
    skel, sep = estimate_skeleton(data, **kwargs)
    cpdag = estimate_cpdag(skel, sep)
    cpdag = nx.relabel.relabel_nodes(cpdag, dict(zip(range(len(data.columns)), data.columns)))
    edges = {(u, v) for u, v in cpdag.edges}
    for u, v in edges:
        if (v, u) in edges:
            cpdag.remove_edge(v, u)
    prior = DAG(cpdag)
    adj_matrix = networkx.to_numpy_array(prior)
    nodes = prior.nodes
    genome = adj_to_genome(adj_matrix)


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    pcpso(data)
