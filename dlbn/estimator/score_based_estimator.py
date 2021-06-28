#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   score_based_estimator.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/6/28 15:11  
------------      
"""
class ScoreBasedEstimator:
    def __init__(self,score_function,search_strategy,data):
        self.score_function = score_function
        self.search_strategy = search_strategy
    def fit(self):
        pass

