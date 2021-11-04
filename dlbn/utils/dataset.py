#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/2 12:27  
------------      
"""
import pandas as pd
from pandas import DataFrame

from dlbn.utils.models import *


class Dataset:
    def __init__(self, name: str, n: int = None):
        self.name = name
        self.load_data(name)
        self.n = n

    @classmethod
    def load_data(cls, name: str):
        path_dir = {
            "asian": r"../../datasets/Asian.csv",
            "cancer": r"../../datasets/cancer.csv"

        }
        return pd.read_csv(path_dir[name])

    @classmethod
    def generate_data(cls, d, n, noise_type='Gaussian', save=False):
        """

        :param d: 邻接矩阵的维度 M = d*d
        :param n: 样本数目 n
        :param noise_type: 'Gaussian'/'non-Gaussian'
        :param save:
        :return:
        data:pd.DataFrame
        dag:np.ndarray
        """
        model = Linear_acyclic_model()
        np_data, dag = model.run(d, n, noise_type, save)
        data = pd.DataFrame(np_data)
        return data, dag


if __name__ == '__main__':
    data = Dataset.load_data("asian")
    print(data)
    data,dag = Dataset.generate_data(6,100)
    print(dag)
