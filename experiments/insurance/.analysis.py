#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   .analysis.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/3/9 9:37  
------------      
"""
import os
import sys
import pandas as pd

sys.path.append("../../")
from bnsl.graph import DAG, compare

cwd = os.getcwd()
print(cwd)
true_dag = DAG()
true_dag.read(r"../data/Insurance/insurance_net.csv",mode='adjacent_matrix')

result_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'SHD', 'norm'])

for n in [50, 500, 2000, 5000]:
    for algorithm in ["ga", "hc", "mmhc", "tabu", "kbnl", "kbnl_c"]:
        g = DAG()
        path = cwd + os.sep + "result" + os.sep + f"{algorithm}+insurance{n}.csv"
        g.read(path, mode='adjacent_matrix')
        res = compare(g, true_dag)
        result_df.loc[f"{algorithm}-{n}"]=pd.Series(res)
print(result_df)
result_df.to_csv("insurance_analysis1.csv")