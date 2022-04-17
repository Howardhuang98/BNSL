#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   expert.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/1 19:07  
------------      
"""
import numpy as np
import pandas as pd
import warnings

from bnsl.graph import DAG

"""
用表格的方式来记录专家知识
专家知识的表达形式：
行指向列!
     A    B    C    D   
A  0     0.1  0.5  0.3
B  0.8   0    0.2  0.2
C  0.7   0.3  0    0.1
D  0.3   0.9  0.1  0
"""


class Expert:
    def __init__(self, expert_data=[pd.DataFrame], expert_confidence=None):
        """
        Expert class

        Args:
            expert_data: a list of experts knowledge matrix[pd.DataFrame].
            expert_confidence: a list of confidence.
        """
        if expert_confidence is None:
            expert_confidence = [1]
        if expert_data is None:
            expert_data = []
        assert len(expert_data) == len(expert_confidence)
        self.expert_data = expert_data
        self.variables = self.expert_data[0].columns
        self.num_expert = len(expert_data)
        # check every matrix in the iterator
        for e in expert_data:
            if e.columns.all() != e.index.all():
                raise ValueError("Expert matrix is wrong, check column and index!")
            # force the diagonal element to 0
            e.values[tuple([np.arange(e.shape[0])] * 2)] = 0
            for i in range(len(e.columns)):
                for j in range(i):
                    if e.values[i, j] + e.values[j, i] > 1:
                        warnings.warn("Opinions added together cannot exceed 1! "
                                      f"position: [{i},{j}] [{j},{i}]with value:{e.values[i, j]},{e.values[i, j]}, all of them "
                                      f"will be corrected into 1/3")
                        e.iloc[i, j] = 1 / 3
                        e.iloc[j, i] = 1 / 3

        self.fused_matrix = self.expert_data[0].copy()
        self.fused_matrix.iloc[:] = 0
        for i in range(self.num_expert):
            for u in self.fused_matrix.columns:
                for v in self.fused_matrix.columns:
                    if u == v:
                        self.fused_matrix.loc[u, v] = 0
                    else:
                        self.fused_matrix.loc[u, v] += expert_confidence[i] * self.expert_data[i].loc[u, v]

    def think(self, u, v):
        """
        the fused opinion based on fused matrix
        :param u: string of node
        :param v: string of node
        :return: [u->v, u<-v, no edge between u and v]
        """
        situation1 = self.fused_matrix.loc[u, v]
        situation2 = self.fused_matrix.loc[v, u]
        situation3 = 1 - situation1 - situation2
        return [situation1, situation2, situation3]

    def norm_distance(self, ground_truth: DAG):
        """
        Calculate the accuracy of the expert's fused matrix.

        Args:
            ground_truth: the ground truth DAG instance.

        Returns:
            Accuracy value.

        """
        diff = self.fused_matrix.values - ground_truth.adj_matrix
        return np.linalg.norm(diff)


if __name__ == '__main__':
    a = np.matrix([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]]
    )
    a1 = np.matrix([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]]
    )
    a2 = np.matrix([
        [0, 0, 1/3],
        [0, 0, 0],
        [0, 0, 0]]
    )
    b = np.matrix([
        [0, 1 / 3, 1 / 3],
        [1 / 3, 0, 1 / 3],
        [1 / 3, 1 / 3, 0]]
    )
    c = np.matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]
    )
    d = np.array([1, 0, 0])
    e = np.array([1 / 3, 1 / 3, 1 / 3])
    f = np.array([0, 0, 1])
    print(np.sum((a-c)*(a-c)))
    print(np.linalg.norm(a1 - c))
    print(np.linalg.norm(a2 - c))
    print(np.linalg.norm(b-c))
    print(np.linalg.norm(c-c))
