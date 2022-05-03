#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   .s.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/5/3 13:46  
------------      
"""
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("alarm.csv", index_col=0)
    for s in data:
        vals = pd.unique(data[s])
        print(vals)
        dic_map = {}
        for k, v in zip(vals, range(0, len(vals))):
            dic_map[k] = v
        for i in range(len(data[s])):
            data.loc[i, s] = dic_map[data.loc[i, s]]
        print(pd.unique(data[s]))

    data.to_csv("alarm_.csv")
