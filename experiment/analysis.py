#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   analysis.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/5/3 14:34  
------------      
"""
import re
import os
import sys

sys.path.append('../')

from pathlib import Path
import bnsl

import pandas as pd


def analyze():
    acc = pd.DataFrame()
    pre = pd.DataFrame()
    rec = pd.DataFrame()
    shd = pd.DataFrame()
    nor = pd.DataFrame()
    f1 = pd.DataFrame()

    path = Path(__file__)
    asia = bnsl.Dataset('asian')
    alarm = bnsl.Dataset('alarm')
    insurance = bnsl.Dataset('insurance')
    hailfinder = bnsl.Dataset('hailfinder')
    ground_truth = {
        'asia': asia.dag,
        'alarm': alarm.dag,
        'insurance': insurance.dag,
        'hailfinder': hailfinder.dag
    }
    for file in Path(path.parent / "result").iterdir():
        network, n, algo = file.stem.split("-")
        g = bnsl.DAG()
        g.read(file, mode="adjacent_matrix")
        mess = bnsl.graph.compare(g, ground_truth[network])
        acc.loc[algo,network+'-'+n]=mess['accuracy']
        pre.loc[algo,network+'-'+n]=mess['precision']
        rec.loc[algo, network + '-' + n] = mess['recall']
        shd.loc[algo, network + '-' + n] = mess['SHD']
        nor.loc[algo, network + '-' + n] = mess['norm']
        f1.loc[algo, network + '-' + n] = mess['f1']

    acc.to_csv(r"./reports/acc.csv")
    pre.to_csv(r"./reports/pre.csv")
    rec.to_csv(r"./reports/rec.csv")
    shd.to_csv(r"./reports/shd.csv")
    nor.to_csv(r"./reports/nor.csv")
    f1.to_csv(r"./reports/f1.csv")




if __name__ == '__main__':
    analyze()
