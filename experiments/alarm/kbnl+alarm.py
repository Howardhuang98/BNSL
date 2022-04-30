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

import pandas as pd

path = pathlib.Path(__file__)
sys.path.append(path.parent.parent.parent.resolve().as_posix())
print(sys.path)

from bnsl.estimators import KBNL


def func():
    for n in [500]:
        print(f"****{n}****")
        data = pd.read_csv(r"../data/Alarm/alarm.csv")[:n]
        expert = [
            r"./result/ga+alarm{}.csv".format(n),
            r"./result/hc+alarm{}.csv".format(n),
            r"./result/mmhc+alarm{}.csv".format(n),
            r"./result/tabu+alarm{}.csv".format(n),
        ]
        kbnl = KBNL(data, expert, [0.25, 0.25, 0.25, 0.25])
        kbnl.run(restart=8, explore_num=8)


if __name__ == '__main__':
    func()
