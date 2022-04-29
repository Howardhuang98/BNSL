#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   score.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/9/6 14:56  
------------      
"""
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.special import gammaln
from .base import Score
from .expert import Expert


class MDL_score(Score):
    """
    MDL score.

    reference:

        Lam, W., & Bacchus, F. (1994). Learning Bayesian belief networks: An approach based on the MDL principle. Computational intelligence, 10(3), 269-293.
        Yuan, C., Malone, B., & Wu, X. (2011, June). Learning optimal Bayesian networks using A* search. In Twenty-Second International Joint Conference on Artificial Intelligence.
    """

    def __init__(self, data: pd.DataFrame):
        super(MDL_score, self).__init__(data)

    def likelihood(self, Nijk: np.ndarray):
        """
        Calculate the likelihood.
        ::
            Nijk is a table-like array.
                q1  q2  q3
            r1  10  12  30
            r2  50  60  17
            (qi are the states of parents)
            (ri are the states of x)

        Args:
            Nijk: state table
        Returns:
            likelihood.
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

    @lru_cache(int(1e6))
    def local_score(self, x: str, parents: tuple):
        """
        Calculate local score.

        References:
            Kitson, N.K., Constantinou, A.C., Guo, Z., Liu, Y. and Chobtham, K., 2021. A survey of Bayesian Network structure learning. arXiv preprint arXiv:2109.11415.

        Args:
            x: str: Node.
            parents: tuple: Parents of node.

        Returns:
            float: Local score
        """

        state_count = self.state_count(x, list(parents))
        # if the state_count has 0 in the array, it will old_result numerical error in log(), to
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
    BIC score class, BIC score is minus MDL score.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(BIC_score, self).__init__(data)
        self.mdl = MDL_score(data)

    def local_score(self, x, parents):
        """
        Calculate local score of BIC score.

        Args:
            x: str: node
            parents: tuple: parents of node.

        Returns:
            BIC score
        """
        score = - self.mdl.local_score(x, parents)
        return score


class L2C_score(Score):
    def __init__(self, data: pd.DataFrame, K: pd.DataFrame):
        super(L2C_score, self).__init__(data)
        self.data = data
        self.n = len(data)
        self.node_list = list(self.data.columns)
        self.K = K
        if len(set(self.data.columns) - set(self.K.columns)) != 0 or len(
                set(self.K.columns) - set(self.data.columns)) != 0:
            raise ValueError("Observed data and K do not consist with nodes")
        self.bic = BIC_score(data)

    def local_score(self, x, parents):
        first_second_part = self.bic.local_score(x, parents)
        p = []
        for node in self.node_list:
            thinks = [self.K.loc[node][x],self.K.loc[x][node],1-self.K.loc[node][x]-self.K.loc[x][node]]
            if node in parents:
                p.append(thinks[0])
            elif node == x:
                continue
            else:
                p.append(thinks[1]+thinks[2])

        third_part = np.log(self.n) * self.activation_function(sum(p))
        return first_second_part + third_part

    def activation_function(self, x):
        l = len(self.node_list)
        zero_point = (1 / 3) * (l - 1)
        if x < zero_point:
            y = 0
        else:
            y = 10 * (x - zero_point)
        return y


class Knowledge_fused_score(Score):
    """
    Knowledge fused score where score = likelihood + log p(G|E) + log p(G).
    """

    def __init__(self, data: pd.DataFrame, expert: Expert):
        super(Knowledge_fused_score, self).__init__(data)
        self.mdl = MDL_score(data)
        self.expert = expert
        self.activation_parameter = []
        self.n = data.shape[0]

    @lru_cache(int(1e5))
    def local_score(self, x: str, parents: tuple):
        """
        Calculate local score.

        Args:
            x: node.
            parents: parents of node.

        Returns:
            Knowledge fused score.

        """
        likelihood = - self.mdl.local_score(x, parents)
        log_pg = np.log(self.n) * self.activation_function(self.multiply_epsilon(x, parents))
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

    def activation_function(self, x):
        n = len(self.data.columns)
        zero_point = (1 / 3) * (n - 1)
        if x < zero_point:
            y = 0
        else:
            y = 10 * (x - zero_point)
        return y


class BDeu_score(MDL_score):
    """
    BDeu score.
    """

    def __init__(self, data: pd.DataFrame, equivalent_sample_size=10, **kwargs):
        super(BDeu_score, self).__init__(data)
        self.equivalent_sample_size = np.float_(equivalent_sample_size)

    def local_score(self, x, parents):
        """
        Calculate local score.

        Args:
            x: node.
            parents: parents of node.

        Returns:
            BDeu score.

        """
        parents = list(parents)
        state_count = self.state_count(x, parents)
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
