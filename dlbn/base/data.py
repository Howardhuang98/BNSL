#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/25 14:50
------------
"""

import pandas as pd


class Data:
    """
    Data类，用于数据处理
    """

    def __init__(self, DataFrame: pd.DataFrame = None):
        self.DataFrame = DataFrame
        self.variables = list(DataFrame.columns.values)
        self.state_names = dict()
        for var in self.variables:
            self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable):
        """Return a list of states that the variable takes in the data."""
        states = sorted(list(self.DataFrame.loc[:, variable].unique()))
        return states

    def _state_count(self, **kwargs):
        """
        状态计数器，用于计算特定的条件下的样本数
        parameters:
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

    def contingency_table(self, **kwargs):
        pass


if __name__ == '__main__':
    data = pd.read_excel(r"../test/test_data.xlsx")
    a = Data(data)
    print(a.variables)
    print(a._state_count(A=1, B=0))
