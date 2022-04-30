#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   kbnl+asia.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/5 11:15  
------------      
"""
import pathlib
import sys

path = pathlib.Path(__file__)
sys.path.append(path.parent.parent.parent.resolve().as_posix())
print(sys.path)
import pandas as pd

from bnsl.estimators import KBNL


def func():
    for n in [50, 500, 2000, 5000]:
        print(f"****{n}****")
        data = pd.read_csv(r"../data/Alarm/alarm.csv")[:n]
        expert = [
            r"./result/ga+alarm{}.csv".format(n),
            r"./result/hc+alarm{}.csv".format(n),
            r"./result/mmhc+alarm{}.csv".format(n),
            r"./result/tabu+alarm{}.csv".format(n),
        ]
        kbnl = KBNL(data, expert, [0.05, 0.2, 0.55, 0.2])
        kbnl.run_parallel(worker=30, restart=15, explore_num=8)
        kbnl.result.to_csv_adj(r"./result/kbnl_cc+alarm{}.csv".format(n))


if __name__ == '__main__':
    func()
