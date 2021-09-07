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
from numpy import log

from DAG import DAG


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


class MDL_score(Score):
    """
    MDL score

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
        column_index = pd.MultiIndex.from_product(parents_states, names=parents)
        state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
        return state_counts

    def local_score(self,x:str,parents:list):
        """
        calculate local score
        参考：
        pgmpy代码；
        《Learning Optimal Bayesian Networks Using A* Search》
        :param x:
        :param parents:
        :return: score
        """
        state_count = self.state_count(x,parents)
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
        K = (x_cardinality-1)*num_parents_states
        score = H + np.log(sample_size)/2 * K
        return score



if __name__ == '__main__':
    dag = DAG()
    s = MDL_score(pd.read_csv(r"../datasets/Asian.csv"), dag)
    print(s.local_score('smoke', ['tub', 'lung']))
