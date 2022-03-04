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

    def likelihood(self, Nijk: np.ndarray):
        """

        Args:
            Nijk:
            while there the parent set is not None, Nijk is a table
                q1  q2  q3
            r1  10  12  30
            r2  50  60  17
            (qi are the states of parents)
            (ri are the states of variable)

            while the parent is None, Nijk is a 1 dim array
                r1  r2
            q   50  60


        Returns:

        """
        # There is parent
        if Nijk.ndim > 1:
            # the log condition array is an 1 * r
            Nij = np.sum(Nijk, axis=0, dtype=np.float_)
            likelihood = np.sum(np.log(Nijk / Nij) * Nijk)
        # Parent is None
        else:
            Nij = sum(Nijk)
            a = Nijk / Nij
            likelihood = np.sum(np.log(Nijk / Nij) * Nijk)
        return likelihood

    def local_score(self, x: str, parents: list):
        """
        calculate local score.
        MDL score = -Likelihood + log N / 2 * F.

        reference:
        pgmpy代码；
        《Learning Optimal Bayesian Networks Using A* Search》
        :param x: name of node
        :param parents: list of parents
        :return: score
        """
        state_count = self.state_count(x, parents)
        # if the state_count has 0 in the array, it will result numerical error in log(), to
        # avoid this error, add 1 on each 0 value
        state_count[state_count == 0] = 1
        Nijk = np.asarray(state_count)
        likelihood = self.likelihood(Nijk)
        if Nijk.ndim > 1:
            q = float(Nijk.shape[1])
            r = float(Nijk.shape[0])
        else:
            q = 1
            r = float(Nijk.shape[0])
        F = (r - 1) * q
        score = -likelihood + np.log(np.sum(Nijk)) / 2 * F
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
        log_pg = np.log(self.n) * self.activation_function(self.multiply_epsilon(x, parents), activation='else')
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
                E += thinks[2] + thinks[1]
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
                y = 10 * (x - zero_point)
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
        Nijk = np.asarray(state_count, dtype=np.float_)
        Nij = np.sum(Nijk, axis=0, dtype=np.float_)
        if parents:
            r = np.float_(len(state_count.index))
            q = np.float_(len(state_count.columns))
        else:
            r = np.float_(len(state_count))
            q = np.float_(1)
        second_term = Nijk + self.equivalent_sample_size / (r * q)
        second_term = gammaln(second_term) - gammaln(self.equivalent_sample_size / (r * q))
        second_term = np.sum(second_term)

        first_term = gammaln(self.equivalent_sample_size / q) - gammaln(Nij + self.equivalent_sample_size / q)
        first_term = np.sum(first_term)
        score = first_term + second_term
        return score


if __name__ == '__main__':
    pass
