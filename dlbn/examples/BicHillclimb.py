#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BicHillclimb.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/29 13:14  
------------      
"""
import dlbn
from dlbn import *
data = pd.read_csv(r"../datasets/Asian.csv")
hc = dlbn.estimator.HillClimb(BicScore,data,max_iter=100)
result_dag = hc.run()
print(result_dag.edges)
