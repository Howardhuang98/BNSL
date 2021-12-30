#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bionics.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/27 22:14  
------------      
"""
from copy import copy
import networkx as nx
from numpy.random import permutation
from tqdm import tqdm
from dlbn.graph import DAG
from dlbn.score import *


class Genetic:
    def __init__(self, data: pd.DataFrame, score_method=BIC_score, pop=40, max_iter=150, c1=0.5, c2=0.5, w=0.05):
        """
        X is a np.array [pop,dim_genome]
                genome
        pop1    01010101
        pop2
        pop3

        """
        self.max_iter = max_iter
        self.score_method = score_method
        self.data = data
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pop = pop
        self.dim_genome = len(data.columns) ** 2
        self.X = np.zeros(shape=(pop, self.dim_genome))
        self.dag = DAG()
        self.personal_best_solution = copy(self.X)
        self.global_best_solution = None
        self.best_X = copy(self.X)
        self.history = []

    def generate_gen(self):
        g = DAG()
        nodes = permutation(list(self.data.columns))
        for i in range(len(nodes)):
            v = nodes[i]
            num_parents = np.random.randint(0,len(nodes)-i)
            parent_list = permutation(nodes[i+1:])[:num_parents]
            for pa in parent_list:
                g.add_edge(pa,v)
            if parent_list.size == 0:
                g.add_node(v)
        return g.genome


    def mutate(self):
        mutated_X = copy(self.X)
        for i in range(self.X.shape[0]):
            if self.has_cycle(self.X[i]) or np.random.rand(1) < self.w:
                mutated_X[i] = self.generate_gen()
        return mutated_X

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
        scores = []
        for genome in self.X:
            self.dag.from_genome(genome, self.data.columns)
            s = self.dag.score(self.score_method, self.data)
            scores.append(s)
        return np.asarray(scores)

    def get_global_best_position(self, pop_score, X):
        idx = np.argmax(pop_score)
        return X[idx]

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
        personal_best_position = self.X
        global_best_position = self.get_global_best_position(pop_score, personal_best_position)
        for i in tqdm(range(self.max_iter)):
            self.history.append(self.X)
            self.X = self.crossover(self.X, personal_best_position, self.c1)
            self.X = self.crossover(self.X, np.expand_dims(global_best_position, axis=0).repeat(self.pop, axis=0),
                                    self.c2)
            self.X = self.mutate()
            new_pop_score = self.pop_score()
            for j in range(self.pop):
                if new_pop_score[j] > pop_score[j]:
                    personal_best_position[j] = self.X[j]
            global_best_position = self.get_global_best_position(pop_score, personal_best_position)

        return global_best_position, np.asarray(self.history)


if __name__ == '__main__':
    data = pd.read_csv(r"../datasets/Asian.csv")
    pso = Genetic(data, pop=40,max_iter=200)
    solu, history = pso.run()
    g = DAG()
    g.from_genome(solu,data.columns)
    print(g.edges)
    g.show(BIC_score,data)
    print(pso.has_cycle(solu))
