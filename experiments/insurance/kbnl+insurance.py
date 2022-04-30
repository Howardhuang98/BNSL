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
    data = pd.read_csv(r"../data/Insurance/insurance.csv")[:n]
    expert = [
        r"./result/ga+insurance{}.csv".format(n),
        r"./result/hc+insurance{}.csv".format(n),
        r"./result/mmhc+insurance{}.csv".format(n),
        r"./result/tabu+insurance{}.csv".format(n),
    ]
    kbnl = KBNL(data, expert, [0.25, 0.25, 0.25, 0.25])
    kbnl.run(restart=50)
    kbnl.result.to_csv_adj(r"./result/kbnl+insurance{}.csv".format(n))
