#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   l2c+alarm.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/30 14:54  
------------      
"""
import pathlib
import sys
path = pathlib.Path(__file__)
sys.path.append(path.parent.parent.parent.resolve().as_posix())
print(sys.path)

import pandas as pd
import bnsl


def func():
    for n in [500]:
        print(f"****{n}****")
        data = pd.read_csv(r"../data/Alarm/alarm.csv")[:n]
        expert = [
            pd.read_csv(r"./result/ga+alarm{}.csv".format(n),index_col=0),
            pd.read_csv(r"./result/ga+alarm{}.csv".format(n),index_col=0),
            pd.read_csv(r"./result/ga+alarm{}.csv".format(n),index_col=0),
            pd.read_csv(r"./result/ga+alarm{}.csv".format(n),index_col=0),
        ]
        l2c = bnsl.L2C(data, expert, [0.25, 0.25, 0.25, 0.25])
        l2c.run()
        print(l2c.result.edges)


if __name__ == '__main__':
    func()