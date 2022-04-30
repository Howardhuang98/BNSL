#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/30 10:54  
------------      
"""
from pathlib import Path

import pandas as pd

from bnsl import DAG

path = Path(__file__).parent

file_dir = {
    'asian': path / 'datasets' / 'asian',
    'cancer': path / 'datasets' / 'cancer',
    'alarm': path / 'datasets' / 'alarm',
    'child': path / 'datasets' / 'child',
    'hailfinder': path / 'datasets' / 'hailfinder',
    'insurance': path / 'datasets' / 'insurance',
}


class Dataset:
    def __init__(self, name):
        self.name = name
        self.file_path = file_dir[name]
        self.data = pd.read_csv(self.file_path/(name+".csv"),index_col=0)
        self.dag = DAG()
        self.dag.read(self.file_path/(name+"_net.csv"))

    def __repr__(self):
        s = f"=====Summary=====\n" \
            f"Data path: {self.file_path.resolve()}\n" \
            f"{self.data.head(3)}\n" \
            f"DAG has {len(self.dag.nodes)} nodes, {len(self.dag.edges)} edges.\n"
        return s
