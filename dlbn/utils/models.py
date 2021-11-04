#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   models.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/10/24 12:22  
------------      
"""
from copy import deepcopy

import numpy as np


class Linear_acyclic_model:
    # use to generate synthetic data
    # based on https://github.com/cdt15/lingam
    def __init__(self):
        pass

    def generate_W(self, d=6, prob=0.7, low=0.5, high=2.0):
        """
        随机生成一个邻接矩阵
        :param d:
        :param prob:
        :param low:
        :param high:
        :return:
        """
        g_random = np.float32(np.random.rand(d, d) < prob)
        g_random = np.tril(g_random, -1)
        U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
        U[np.random.randn(d, d) < 0] *= -1
        W = (g_random != 0).astype(float) * U
        return W

    def generate_data(self, W: np.ndarray, n, noise_type='Gaussian', permutate=True):
        """
        x = Wx+e
        the model described in 《A Linear Non-Gaussian Acyclic Model for Causal Discovery》
        :param noise_type:
        :param W:
        :param n: 生成数据 X(d*n)
        :return:

        """
        d = W.shape[0]
        if noise_type == 'Gaussian':
            e_std = np.random.uniform(0.5, 2, d)
            e = np.random.normal(np.zeros((n, d)), e_std)
        if noise_type == 'non-Gaussian':
            q = np.random.rand(d) * 1.1 + 0.5
            ixs = np.where(q > 0.8)
            q[ixs] = q[ixs] + 0.4

            # Generates disturbance variables
            e = np.random.randn(n, d)
            e = np.sign(e) * (np.abs(e) ** q)

            # Normalizes the disturbance variables to have the appropriate scales
            e_std = np.random.uniform(0.5, 2, d)
            e = e / np.std(e, axis=0) * e_std

        x = np.zeros((n, d))
        c = np.zeros(d)
        for i in range(d):
            x[:, i] = x.dot(W[i, :]) + e[:, i] + c[i]
        if permutate:
            W_ = deepcopy(W)
            c_ = deepcopy(c)
            p = np.random.permutation(d)
            x[:, :] = x[:, p]
            W_[:, :] = W_[p, :]
            W_[:, :] = W_[:, p]
            c_[:] = c[p]
        return x, W_, c_

    def run(self, d, n, noise_type, save=True):
        seed = np.random.randint(1, 100)
        np.random.seed(seed)
        w = self.generate_W(d=d)
        data, dag, c = self.generate_data(W=w, n=n, noise_type=noise_type)
        if save:
            np.save('data-d={}-seed{}.npy'.format(d, seed), data)
            np.save('dag-d={}-seed{}.npy'.format(d, seed), dag)
        return data, dag


if __name__ == '__main__':
    dg = Linear_acyclic_model()
    a, b = dg.run(6, 100, 'non-Gaussian', save=False)
    print(a)
