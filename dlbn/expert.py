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
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = data.columns
        # 此处最好能做一个检查，确保values[i,j]+values[j,i]<=1
        #
        #     待补充
        #
        if data.columns.all() != data.index.all():
            raise ValueError("Expert matrix is wrong, check column and index!")
        # force the diagonal element to 0
        self.data.values[tuple([np.arange(self.data.shape[0])] * 2)] = 0
        for i in range(len(self.data.columns)):
            for j in range(i):
                if self.data.values[i, j] + self.data.values[j, i] > 1:
                    raise ValueError(
                        "Opinions added together cannot exceed 1! "
                        "position: [{},{}] with value:{},{}".format(i, j, self.data.values[i, j],self.data.values[j, i]))

    def think(self, u, v):
        """
        专家对于u->v，u<-v,u><v的三个概率
        :param u:
        :param v:
        :return:
        """
        situation1 = self.data.loc[u][v]
        situation2 = self.data.loc[v][u]
        situation3 = 1 - situation1 - situation2
        return [situation1, situation2, situation3]


if __name__ == '__main__':
    chen_data = pd.DataFrame({
        "A": [0, 0.8, 0.7, 0.3],
        "B": [0.1, 0, 0.9, 0.9],
        "C": [0.3, 0.05, 0, 0.1],
        "D": [0.3, 0.1, 0.1, 0]
    }, index=["A", "B", "C", "D"])
    chen = Expert(data=chen_data)
    print(chen.think("A", "B"))
