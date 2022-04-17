#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bionics.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/27 22:14  
------------      
"""
import itertools as it
import sys

import networkx as nx
import numpy as np
import numpy.random
import pandas as pd
from numpy.random import permutation
from tqdm import tqdm

from .graph import DAG
from .score import *


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
    def __init__(self, data: pd.DataFrame, num_parent=5, score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5,
                 w=0.05, patience=20):
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

        self.manager_list = pd.DataFrame(columns=["genome", "score", "rank"])
        self.patience = patience
        self.history = []
        self.result = None

    def initialize_manager_list(self):
        """
        update manager list
        :return:
        """
        X = np.random.randint(0, 2, (self.pop, self.dim_genome))
        for i in range(len(X)):
            print("\rInitializing population...{}/{}".format(i, self.pop), end="")
            sys.stdout.flush()
            genome = genome_to_str(X[i])
            self.manager_list.reindex()
            self.manager_list.loc[i, "genome"] = genome
        self.local_optimizer()
        self.manager_list["rank"] = self.manager_list.score.rank(ascending=False)
        self.manager_list.sort_values(by="rank", inplace=True)
        self.manager_list.index = range(self.manager_list.shape[0])

    def legal_parents_iter(self,parents,n):
        n = len(parents)
        num_com = n*(n-1)/2
        count = 0
        while count < n and count <= num_com:
            count += 1
            yield next(it.combinations(parents, self.u))

    def local_optimizer(self):
        """
        input a population, make sure each node's parents do not exceed u.
        :return: a new population
        """
        for i in range(self.manager_list.shape[0]):
            genome = self.manager_list.loc[i, "genome"]
            # for the dag of genome, do local optimization
            dag = genome_to_dag(genome, self.node_order)
            for node in dag.nodes:
                parents = list(dag.predecessors(node))
                if len(parents) > self.u:
                    dag.remove_edges_from([(p, node) for p in parents])
                    current_parents = []
                    current_score = 0
                    # this iteration will have C_n^u times
                    for legal_parents in self.legal_parents_iter(parents, self.pop):
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

            new_genome = genome_to_str(dag_to_genome(dag, self.node_order))
            self.manager_list.loc[i, "genome"] = new_genome
            self.manager_list.loc[i, "score"] = genome_to_dag(new_genome, self.node_order).score(
                self.score_method)

    def mutate(self):
        """
        randomly mutate on one bit position
        :return:
        """
        pass

    def select_parents(self, num_parent):
        selected_list = []
        # probability = (self.pop - self.manager_list["rank"]) / ((self.pop - 1) * self.pop / 2)
        for i in range(num_parent):
            selected = numpy.random.choice(self.manager_list.index.values, 2)
            selected_list.append(list(selected))
        return selected_list

    def produce_children(self, parents_list):
        child = []
        children = pd.DataFrame(columns=["genome", "score", "rank"])
        for i, j in parents_list:
            x = list(self.manager_list.loc[i, "genome"])
            y = list(self.manager_list.loc[j, "genome"])
            for index in range(self.dim_genome):
                if np.random.rand(1) < 0.5:
                    child.append(x[index])
                else:
                    child.append(y[index])
            children.loc[children.shape[0], ["genome", "score"]] = [genome_to_str(child),
                                                                    genome_to_dag(child, self.node_order).score(
                                                                        self.score_method)]
            child = []
        self.manager_list = pd.concat([self.manager_list, children])
        self.manager_list["rank"] = self.manager_list.score.rank(ascending=False)
        self.manager_list.sort_values(by="rank", inplace=True)
        self.manager_list.index = range(self.manager_list.shape[0])

    def reduce_population(self):

        self.manager_list = self.manager_list.iloc[:self.pop]

    def update_order(self):
        self.node_order = np.random.permutation(self.node_order)

    def run(self):
        # update manager list
        print("Initializing population...", end="")
        sys.stdout.flush()
        self.initialize_manager_list()
        print("\rInitializing population---------->Done")
        sys.stdout.flush()
        best_score = -float('inf')
        count = 0
        for i in range(self.max_iter):
            print("\rIterating with best score: {}---------->{}/{}".format(best_score, i, self.max_iter), end="")
            sys.stdout.flush()
            selected_parents = self.select_parents(int(self.pop / 3))
            self.produce_children(selected_parents)
            # self.mutate()
            self.reduce_population()
            self.local_optimizer()
            if 15 % (i + 1) == 0:
                self.update_order()
            current_score = self.manager_list.iloc[0]["score"]
            if current_score > best_score:
                best_score = current_score
            else:
                count += 1
                if count > self.patience:
                    break
            self.history.append(self.manager_list.iloc[0]["score"])
        best_genome = self.manager_list.iloc[0]["genome"]
        self.result = genome_to_dag(best_genome, self.node_order)
        # avoid missing isolated node
        self.result.add_nodes_from(self.data.columns)
        return self.result


if __name__ == '__main__':
    pass
