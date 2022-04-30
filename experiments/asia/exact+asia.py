#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ga+hailfinder.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/3 17:59  
------------      
"""
import pandas as pd

from bnsl.estimators import DP

for n in [50,500,2000,5000]:
    print("****{}****".format(n))
    data = pd.read_csv(r"../data/Asia/Asian.csv")[:n]
    dp = DP(data)
    dag = dp.run()
    dag.to_csv_adj(r"./old_result/exact+asia{}.csv".format(n))