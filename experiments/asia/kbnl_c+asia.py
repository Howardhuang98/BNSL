#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   kbnl+asia.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/5 11:15  
------------      
"""
import pandas as pd

from bnsl.estimators import KBNL

for n in [50, 500, 2000, 5000]:
    print(f"****{n}****")
    data = pd.read_csv(r"../data/Asia/Asian.csv")[:n]
    expert = [
        r"./result/ga+asia{}.csv".format(n),
        r"./result/hc+asia{}.csv".format(n),
        r"./result/mmhc+asia{}.csv".format(n),
        r"./result/tabu+asia{}.csv".format(n),
    ]
    kbnl = KBNL(data, expert, [0.05, 0.25, 0.5, 0.2])
    kbnl.run(restart=50)
    kbnl.result.to_csv_adj(r"./result/kbnl_c+asia{}.csv".format(n))
