#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   score.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/6 14:56  
------------      
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.special import gammaln

from dlbn.base import Score
from dlbn.expert import Expert


class MDL_score(Score):
    """
    MDL score, it must be a positive number. A optimal network is the network with lowest MDL score!

    reference:
    Lam, W., & Bacchus, F. (1994). Learning Bayesian belief networks: An approach based on the MDL principle. Computational intelligence, 10(3), 269-293.
    Yuan, C., Malone, B., & Wu, X. (2011, June). Learning optimal Bayesian networks using A* search. In Twenty-Second International Joint Conference on Artificial Intelligence.
    """

    def __init__(self, data: pd.DataFrame):
        super(MDL_score, self).__init__(data)

    def state_count(self, x: str, parents: list):
        """
        count a multi-index table
        :param x:
        :param parents:
        :return:
        """
        if parents:
            parents_states = [self.state_names[parent] for parent in parents]
            state_count_data = (
                self.data.groupby([x] + parents).size().unstack(parents)
            )
            row_index = self.state_names[x]
            if len(parents) == 1:
                column_index = parents_states[0]
            else:
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
        else:
            state_counts = self.data.groupby(x).size()
        return state_counts

    def likelihood_score(self, x: str, parents: list):
        state_count = self.state_count(x, parents)
        # if the state_count has 0 in the array, it will result numerical error in log(), to
        # avoid this error, add 1 on each 0 value
        state_count[state_count == 0] = 1
        counts = np.asarray(state_count)

        sample_size = len(self.data)
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)
        # the counts data is different in the condition of parents = [] and other
        np.log(counts, out=log_likelihoods, where=counts > 0)
        if parents:
            num_parents_states = float(state_count.shape[1])
            # the log condition array is an 1 * r
            log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
            np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        else:
            num_parents_states = 1
            # the log conditionals is a float
            log_conditionals = np.log(np.sum(counts, axis=0, dtype=np.float_))
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        # sum
        likelihood = np.sum(log_likelihoods)
        return likelihood

    def local_score(self, x: str, parents: list):
        """
        calculate local score.

        reference:
        pgmpy代码；
        《Learning Optimal Bayesian Networks Using A* Search》
        :param x: name of node
        :param parents: list of parents
        :return: score(x|parents)
        """
        state_count = self.state_count(x, parents)
        # if the state_count has 0 in the array, it will result numerical error in log(), to
        # avoid this error, add 1 on each 0 value
        state_count[state_count == 0] = 1
        counts = np.asarray(state_count)

        sample_size = len(self.data)
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)
        # the counts data is different in the condition of parents = [] and other
        np.log(counts, out=log_likelihoods, where=counts > 0)
        if parents:
            num_parents_states = float(state_count.shape[1])
            # the log condition array is an 1 * r
            log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
            np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        else:
            num_parents_states = 1
            # the log conditionals is a float
            log_conditionals = np.log(np.sum(counts, axis=0, dtype=np.float_))
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
        x_cardinality = float(state_count.shape[0])

        # sum
        H = - np.sum(log_likelihoods)
        # K
        K = (x_cardinality - 1) * num_parents_states
        score = H + np.log(sample_size) / 2 * K
        return score


class BIC_score(Score):
    """
    BIC is minus MDL score
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(BIC_score, self).__init__(data)
        self.mdl = MDL_score(data)

    def local_score(self, x, parents):
        score = - self.mdl.local_score(x, parents)
        return score


class Knowledge_fused_score(Score):
    """
    Knowledge fused score
    score = likelihood + log p(G|E)
    where E is fused expert matrix
    """

    def __init__(self, data: pd.DataFrame, expert: Expert):
        super(Knowledge_fused_score, self).__init__(data)
        self.mdl = MDL_score(data)
        self.expert = expert
        self.activation_parameter = []
        self.n = data.shape[0]

    def local_score(self, x, parents):
        # likelihood = self.mdl.likelihood_score(x, parents)
        likelihood = - self.mdl.local_score(x, parents)
        log_pg = np.log(self.n)*self.activation_function(self.multiply_epsilon(x, parents), activation='else')
        return likelihood + log_pg

    def multiply_epsilon(self, x, parents):
        parents = set(parents)
        sample_size = len(self.data)
        # calculate the multiply epsilon
        E = 0
        for node in self.data.columns:
            # thinks = [u->v, u<-v, u><v]
            thinks = self.expert.think(node, x)
            if node == x:
                continue
            elif node in parents:
                E += thinks[0]
            else:
                E += thinks[2]+thinks[1]
        return E

    def activation_function(self, x, activation="else"):
        if activation == "cubic":
            parameters = self.get_activation_parameter()
            y = parameters[0] * x ** 3 + parameters[1] * (x ** 2) + parameters[2] * x + parameters[3]
        if activation == "else":
            n = len(self.data.columns)
            zero_point = (1 / 3) * (n - 1)
            if x < zero_point:
                y = 0
            else:
                y = 10 * (x-zero_point)
        return y

    def get_activation_parameter(self):
        n = len(self.data.columns)
        zero_point = (1 / 3) ** (n - 1)

        def func(i, z):
            a, b, c, d = i[0], i[1], i[2], i[3]
            return [
                a * z ** 3 + b * z ** 2 + c * z + d,
                3 * a * z ** 2 + 2 * b * z + c,
                a + b + c + d - 1000000,
                3 * a + 2 * b + c - 10000000
            ]

        r = fsolve(func, x0=[0, 0, 0, 0], args=zero_point)
        return r

    def show_act(self):
        x = np.arange(0, 1, 0.01)
        y = self.activation_function(x)
        plt.plot(x, y)
        plt.show()


class BDeu_score(MDL_score):
    """
    reference pgmpy code
    https://pgmpy.org/_modules/pgmpy/estimators/StructureScore.html#BDsScore
    """

    def __init__(self, data: pd.DataFrame, equivalent_sample_size=10, **kwargs):
        super(BDeu_score, self).__init__(data)
        self.equivalent_sample_size = np.float_(equivalent_sample_size)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        state_count = self.state_count(variable, parents)
        state_count[state_count == 0] = 1
        counts = np.asarray(state_count, dtype=np.float_)
        Nij = np.sum(counts, axis=0, dtype=np.float_)
        if parents:
            r = np.float_(len(state_count.index))
            q = np.float_(len(state_count.columns))
        else:
            r = np.float_(len(state_count))
            q = np.float_(1)
        second_term = counts + self.equivalent_sample_size / (r * q)
        second_term = gammaln(second_term) - gammaln(self.equivalent_sample_size / (r * q))
        second_term = np.sum(second_term)

        first_term = gammaln(self.equivalent_sample_size / q) - gammaln(Nij + self.equivalent_sample_size / q)
        first_term = np.sum(first_term)
        score = first_term + second_term
        return score


if __name__ == '__main__':
    pass
