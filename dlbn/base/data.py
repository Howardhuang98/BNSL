#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/25 14:50
------------
"""

import itertools

import pandas as pd
from tqdm import tqdm


class Data:
    """
    Data类，用于数据处理
    """

    def __init__(self, DataFrame: pd.DataFrame = None):
        self.DataFrame = DataFrame
        self.variables = list(DataFrame.columns.values)
        self.state_names = dict()
        for var in self.variables:
            self.state_names[var] = self.collect_state_names(var)

    def collect_state_names(self, variable):
        """Return a list of states that the variable takes in the data.
        'A'->[1,0]
        ['A','B']->[(1, 0), (2, 1)]  where list[tuple()]
        """
        if isinstance(variable, str):
            states = sorted(list(self.DataFrame.loc[:, variable].unique()))
            return states
        if isinstance(variable, list):
            list_data = []
            for i, s in self.DataFrame[variable].iterrows():
                s = tuple(s.values.tolist())
                list_data.append(s)
            states = list(set(list_data))
            return states

    def state_count(self, **kwargs):
        """
        状态计数器，用于计算特定的条件下的样本数
        parameters:
        dict: {'A':1, 'B': 2}
        for example (A = 1, B = 2)
        """
        query = kwargs
        count = 0
        for row_tuple in self.DataFrame.iterrows():
            series = row_tuple[1]
            flag = True
            for _ in query.items():
                if series[_[0]] == _[1]:
                    continue
                else:
                    flag = False
                    break
            if flag:
                count += 1

        return count

    def state_count_in_conTB(self, **kwargs):
        query = kwargs
        columns = list(query.keys())
        indexes = []
        for index, series in self.DataFrame.loc[:, columns].iterrows():
            flag = True
            for _ in query.items():
                if series[_[0]] == _[1]:
                    continue
                else:
                    flag = False
                    break
            if flag:
                indexes.append(index)
        count = self.DataFrame.loc[indexes, 'count'].sum()

        return count

    def contingency_table(self, save_dir:str = None):
        """
        计算该data的contingency table: pd.Dataframe
        """
        con_tb = pd.DataFrame(columns=list(self.variables) + ['count'])
        conditions = {}
        for var in self.variables:
            conditions[var] = self.collect_state_names(var)
        for condition in tqdm(itertools.product(*conditions.values()),desc="Generating the contingency table"):
            condition = dict(zip(self.variables, condition))
            condition['count'] = self.state_count(**condition)
            con_tb.loc[con_tb.shape[0]] = condition
        con_tb = con_tb[con_tb['count'] != 0]
        con_tb.reset_index(drop=True, inplace=True)
        if save_dir:
            con_tb.to_excel(save_dir)
        return con_tb

    def AD_tree(self):
        """
        采用AD_tree数据结构进行储存contingency_table
        """
        pass


if __name__ == '__main__':
    data = pd.read_excel(r"../test/test_data.xlsx")
    a = Data(data)
    print(a.collect_state_names(['A', 'B']))
    print(a.state_count(A=1, B=0))
    print(a.contingency_table())
    b = Data(a.contingency_table())
    config = {'B': 0, 'C': 0}
    print(b.state_count_in_conTB(**config))
