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

from dlbn.direct_graph import DAG


class Score:
    """
    Score base class
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.state_names = {}
        for var in list(data.columns.values):
            self.state_names[var] = sorted(list(self.data.loc[:, var].unique()))
        self.contingency_table = None

    def local_score(self, *args):
        return 0


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
        return state_counts

    def local_score(self, x: str, parents: list):
        """
        calculate local score
        参考：
        pgmpy代码；
        《Learning Optimal Bayesian Networks Using A* Search》
        :param x:
        :param parents:
        :return: score
        """
        # if parents list is an empty list
        if not parents:
            return 10e10
        # else
        else:
            state_count = self.state_count(x, parents)
            sample_size = len(self.data)
            num_parents_states = float(state_count.shape[1])
            x_cardinality = float(state_count.shape[0])
            counts = np.asarray(state_count)
            log_likelihoods = np.zeros_like(counts, dtype=np.float_)
            # 求 log N(X,Pa) --> log_likelihoods
            np.log(counts, out=log_likelihoods, where=counts > 0)
            # 求 log N(X) --> log_conditionals
            log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
            np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
            # log (N(X,Pa)/N(X))
            log_likelihoods -= log_conditionals
            # N(X,Pa) * log (N(X,Pa)/N(X))
            log_likelihoods *= counts
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

    def __init__(self, data: pd.DataFrame):
        super(BIC_score, self).__init__(data)
        self.mdl = MDL_score(data)

    def local_score(self, x, parents):
        score = - self.mdl.local_score(x, parents)
        return score


if __name__ == '__main__':
    dag = DAG()
    dag.read_excel(r"../datasets/cancer net.xlsx")
    data = pd.read_csv(r"../datasets/cancer.csv",index_col=0)
    s = dag.score(MDL_score,data)
    print(s)