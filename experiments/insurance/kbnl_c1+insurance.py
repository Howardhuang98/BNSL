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


def run():
    for n in [50]:
        print(f"****{n}****")
        data = pd.read_csv(r"../data/Insurance/insurance.csv")[:n]
        expert = [
            r"./result/ga+insurance{}.csv".format(n),
            r"./result/hc+insurance{}.csv".format(n),
            r"./result/mmhc+insurance{}.csv".format(n),
            r"./result/tabu+insurance{}.csv".format(n),
        ]
        kbnl = KBNL(data, expert, [0.1, 0.45, 0, 0.45])
        kbnl.run_parallel(worker=35, restart=5,explore_num=8, num_parents=5)
        kbnl.result.to_csv_adj(r"./result/kbnl_c1+insurance{}.csv".format(n))


if __name__ == '__main__':
    run()
