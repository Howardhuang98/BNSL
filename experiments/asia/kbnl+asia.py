#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   kbnl+asia.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/5 11:15  
------------      
"""
import pandas as pd

from bnsl.estimators import HC

for n in [50, 500, 2000, 5000]:
    print(f"****{n}****")
    data = pd.read_csv(r"../data/Asia/Asian.csv")[:n]
    expert = [
        r"./old_result/ga+asia{}.csv".format(n),
        r"./old_result/hc+asia{}.csv".format(n),
        r"./old_result/mmhc+asia{}.csv".format(n),
        r"./old_result/tabu+asia{}.csv".format(n),
    ]
    kbnl = HC(data, expert, [0.25, 0.25, 0.25, 0.25])
    kbnl.run(restart=50)
    kbnl.result.to_csv_adj(r"./result/kbnl+asia{}.csv".format(n))
