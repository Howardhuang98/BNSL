#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_process.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/10 18:35  
------------      
"""
import numpy as np
import pandas as pd

data = pd.read_excel(r"./old_result.xlsx")
data1 = data.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
data1["reliability"] = 0
data1["reliability"].iloc[0] = 100
centrality = {"D11": 0.114, "D12": 0.091, "D13": 0.068, "D14": 0.136, "D15": 0.114, "θ": 0.091, "ZX_OUT": 0.068,
              "ZJ_D": 0.068, "ZJ_U": 0.068, "ZX_IN": 0.068, "FL_ZKL": 0.057, "LS_YJL": 0.057, }
def r(data):
    for i in range(1,len(data)):
        s = 0
        for node in centrality.keys():
            weight = centrality[node]
            y = data.iloc[i][node]
            t = data.iloc[0][node]
            s += weight*(y-t)**2
            s = -s
            s = s/25
            s = s/2
            result = 100*(np.exp(s))
            data["reliability"].iloc[i]=result
    return data

data2 = r(data1)
print(data2)