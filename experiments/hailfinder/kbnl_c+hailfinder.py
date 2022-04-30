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
    for n in [5000]:
        print(f"****{n}****")
        data = pd.read_csv(r"../data/Hailfinder/hailfinder.csv")[:n]
        expert = [
            r"./result/ga+hailfinder{}.csv".format(n),
            r"./result/hc+hailfinder{}.csv".format(n),
            r"./result/mmhc+hailfinder{}.csv".format(n),
            r"./result/tabu+hailfinder{}.csv".format(n),
        ]
        kbnl = KBNL(data, expert, [0.05, 0.2, 0.55, 0.2])
        kbnl.run_parallel(worker=36, restart=15, explore_num=12)
        kbnl.result.to_csv_adj(r"./result/kbnl_c+hailfinder{}.csv".format(n))


if __name__ == '__main__':
    run()
