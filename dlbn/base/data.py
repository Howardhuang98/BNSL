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


if __name__ == '__main__':
    data = pd.read_excel(r"../test/test_data.xlsx")
    a = Data(data)
    print(a.variables)
    print(a.state_names)
