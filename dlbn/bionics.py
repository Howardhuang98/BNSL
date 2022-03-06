#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bionics.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/27 22:14  
------------      
"""
import itertools as it

import networkx as nx
import numpy as np
import numpy.random
from numpy.random import permutation
from tqdm import tqdm

from dlbn.graph import DAG
from dlbn.score import *


def genome_to_dag(genome, node_order):
    """
    generate dag with genome
    #todo improve efficiency, do not need to iterate every element in adjacency matrix
    :param genome:
    :param node_order:
    :return:
    """
    dag = DAG()
    genome = list(genome)
    num_nodes = len(node_order)
    adj_list = []
    index = 0
    for n in range(num_nodes - 1, -1, -1):
        num_zeros = num_nodes - n
        adj_list += [0 for _ in range(num_zeros)]
        adj_list += genome[index:index + n]
        index += n
    adj_matrix = np.reshape(np.asarray(adj_list), newshape=(num_nodes, num_nodes))
    adj_matrix = adj_matrix.astype(int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                dag.add_edge(node_order[i], node_order[j])
    return dag


def dag_to_genome(dag, node_order):
    if not isinstance(dag, DAG):
        raise ValueError("input a DAG instance")
    genome = []
    adj = dag.adj_DataFrame(nodelist=node_order).values
    adj = adj.astype(np.int32)
    for i in range(len(node_order)):
        genome += list(adj[i, i + 1:])
    return np.asarray(genome)


def genome_to_str(genome):
    l = ""
    for g in genome:
        l += str(g)
    return l


class Genetic:
    def __init__(self, data: pd.DataFrame, num_parent=4, score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5,
                 w=0.05):
        """
        :arg
        :param data: pd.DataFrame
        :param num_parent: the maximum number of parents

        assume there is an order between nodes, e.g., [a, b, c] means a is a root node without parent.
        the genome is an up triangle matrix.
            a   b    c
        a   0   1,0  1,0
        b   0   0    1,0
        c   0   0    0

        X is a np.array [pop,dim_genome]
                genome
        pop1    01010101
        pop2
        pop3

        """
        self.data = data
        self.max_iter = max_iter
        self.score_method = score_method(self.data)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pop = pop
        self.node_order = list(data.columns)
        self.n = len(data.columns)
        self.u = num_parent
        self.dim_genome = int(self.n * (self.n - 1) / 2)
        # Initialize a population
        self.X = np.random.randint(0, 2, (self.pop, self.dim_genome))
        self.manager_list = pd.DataFrame(columns=["genome", "score", "rank"])

    def update_manager_list(self):
        """
        update manager list
        :return:
        """
        for i in range(len(self.X)):
            score = genome_to_dag(self.X[i], self.node_order).score(self.score_method)
            genome = genome_to_str(self.X[i])
            self.manager_list.loc[i, "genome"] = genome
            self.manager_list.loc[i, "score"] = score
        self.manager_list["rank"] = self.manager_list.score.rank(ascending=False)

    def local_optimizer(self):
        """
        input a population, make sure each node's parents do not exceed u.
        :return: a new population
        """
        for i in range(len(self.X)):
            genome = self.X[i]
            # for the dag of genome, do local optimization
            dag = genome_to_dag(genome, self.node_order)
            for node in dag.nodes:
                parents = list(dag.predecessors(node))
                if len(parents) > self.u:
                    dag.remove_edges_from([(p, node) for p in parents])
                    current_parents = []
                    current_score = 0
                    # this iteration will have C_n^u times
                    for legal_parents in it.combinations(parents, self.u):
                        legal_parents = list(legal_parents)
                        # first running:
                        if not current_parents:
                            current_parents = legal_parents
                            current_score = self.score_method.local_score(node, current_parents)
                        else:
                            new_score = self.score_method.local_score(node, legal_parents)
                            if new_score > current_score:
                                current_parents = legal_parents
                                current_score = new_score
                    dag.add_edges_from([(p, node) for p in current_parents])

            new_genome = dag_to_genome(dag, self.node_order)
            self.X[i] = new_genome

    def mutate(self):
        """
        randomly mutate on one bit position
        :return:
        """
        pop = len(self.X)
        for i in range(pop):
            if np.random.rand(1) < self.w:
                mutation_index = np.random.randint(0, self.dim_genome)
                self.X[i, mutation_index] = int(self.X[i, mutation_index] ^ 1)

    def select_parents(self, num_parent):
        selected_list = []
        probability = (self.pop - self.manager_list["rank"]) / ((self.pop - 1) * self.pop / 2)
        for i in range(num_parent):
            selected = numpy.random.choice(self.manager_list.index.values, 2, p=probability)
            selected_list.append(list(selected))
        return selected_list

    def produce_children(self, parents_list):
        child = []
        children = []
        for i, j in parents_list:
            x = self.X[i]
            y = self.X[j]
            for index in range(self.dim_genome):
                if np.random.rand(1) < 0.5:
                    child.append(x[index])
                else:
                    child.append(y[index])
            children.append(child)
            child = []
        return np.asarray(children)

    def reduce_population(self):
        # ensure the manager list is updated
        assert self.manager_list.shape[0] == self.X.shape[0]
        self.manager_list.sort_values(by="rank",inplace=True)
        self.manager_list = self.manager_list[:self.pop]

    def run(self):
        # update manager list
        self.update_manager_list()
        for i in range(self.max_iter):
            selected_parents = self.select_parents(int(self.pop / 3))
            children = self.produce_children(selected_parents)
            self.mutate()
            self.local_optimizer()
            # concat children genome into X
            self.X = np.vstack((self.X, children))
            # todo, Inefficient, majority of the manager list is unchanged.
            self.update_manager_list()
            self.reduce_population()
        best_genome = self.manager_list.iloc[0]["genome"]
        return genome_to_dag(best_genome, self.node_order)


if __name__ == '__main__':
    pass
