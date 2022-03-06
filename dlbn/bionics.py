#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bionics.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/27 22:14  
------------      
"""
import itertools as it
from copy import copy

import networkx as nx
import numpy as np
from numpy.random import permutation
from tqdm import tqdm

from dlbn.graph import DAG
from dlbn.score import *


def dag_from_genome(genome, node_order):
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
        genome += list(adj[i, i+1:])
    return np.asarray(genome)


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
        self.dag = DAG()
        self.personal_best_solution = copy(self.X)
        self.global_best_solution = None
        self.best_X = copy(self.X)
        self.history = []

    def local_optimizer(self):
        """
        input a population, make sure each node's parents do not exceed u.
        :return: a new population
        """
        for i in range(len(self.X)):
            genome = self.X[i]
            # for the dag of genome, do local optimization
            dag = dag_from_genome(genome, self.node_order)
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

            new_genome = dag_to_genome(dag,self.node_order)
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

    def crossover(self, X1, X2, r):
        children = []
        for i in range(X1.shape[0]):
            if np.random.rand(1) < r:
                child = []
                for j in range(self.dim_genome):
                    if np.random.rand(1) < 0.5:
                        child.append(X1[i, j])
                    else:
                        child.append(X2[i, j])
                children.append(np.asarray(child))
            else:
                children.append(X1[i])
        return np.asarray(children)

    def pop_score(self):
        """
        scores of current X
        :return:
        """
        scores = []
        for genome in self.X:
            self.dag.from_genome(genome, self.data.columns)
            s = self.dag.score(self.score_method, self.data)
            scores.append(s)
        return np.asarray(scores)

    def get_global_best_position(self, pop_score, X):
        idx = np.argmax(pop_score)
        return X[idx]

    def get_global_best_score(self, pop_score):
        idx = np.argmax(pop_score)
        return pop_score[idx]

    def has_cycle(self, gen):
        g = DAG()
        g.from_genome(gen, self.data.columns)
        try:
            cycles = list(nx.find_cycle(g))
        except nx.NetworkXNoCycle:
            return False
        return True

    def run(self):
        pop_score = self.pop_score()
        personal_best_score = pop_score
        personal_best_position = self.X
        global_best_position = self.get_global_best_position(pop_score, personal_best_position)
        global_best_score = 0
        for i in tqdm(range(self.max_iter)):
            if i == 0:
                last_pop_score = pop_score
            else:
                last_pop_score = new_pop_score
            self.X = self.crossover(self.X, personal_best_position, self.c1)
            self.X = self.crossover(self.X, np.expand_dims(global_best_position, axis=0).repeat(self.pop, axis=0),
                                    self.c2)
            self.X = self.mutate()
            new_pop_score = self.pop_score()
            for j in range(self.pop):
                if new_pop_score[j] > last_pop_score[j]:
                    personal_best_position[j] = self.X[j]
                    personal_best_score[j] = new_pop_score[j]

            global_best_position = self.get_global_best_position(new_pop_score, personal_best_position)
            global_best_score = self.get_global_best_score(personal_best_score)
            self.history.append(global_best_score)

        return global_best_position, np.asarray(self.history)


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/asian/Asian.csv")
    pso = Genetic(data, pop=40, max_iter=200)
    solu, history = pso.run()
    g = DAG()
    g.from_genome(solu, data.columns)
    print(g.edges)
    g.show(BIC_score, data)
    print(pso.has_cycle(solu))
