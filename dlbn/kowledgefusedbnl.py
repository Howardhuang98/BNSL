#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   kowledgefusedbnl.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/1/4 11:32  
------------      
"""
import pandas as pd

from dlbn.expert import Expert
from dlbn.graph import DAG
from dlbn.heuristic import HillClimb
from dlbn.score import Knowledge_fused_score


class KowledgeFusedBNL(HillClimb):

    def __init__(self, data: pd.DataFrame, expert_data: list, expert_confidence: list, initial_dag: DAG = None,
                 max_iter=10000, restart=1, explore_num=1):
        if isinstance(expert_data[0], pd.DataFrame):
            self.expert = Expert(expert_data, expert_confidence)
        if isinstance(expert_data[0], str):
            self.expert = Expert.read_excel(expert_data, confidence=expert_confidence)
        super().__init__(data, score_method=Knowledge_fused_score, initial_dag=initial_dag, max_iter=max_iter,
                         restart=restart, explore_num=explore_num, expert=self.expert)
