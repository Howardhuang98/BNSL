#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   generators.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/12/1 14:55  
------------      
"""
import numpy as np
from tqdm import tqdm

from dlbn.dp import *

"""
5-10-15-20 个变量数
1000 样本数
二元离散数据，MDL评分
用动态规划算法求出最段路径
"""


def dp_generate(num_of_nodes=6):
    np_data = np.random.randint(low=0, high=2, size=(1000, num_of_nodes))
    data = pd.DataFrame(data=np_data, columns=[str(i) for i in range(0, num_of_nodes)])
    pg = generate_parent_graph(data, MDL_score)
    og = generate_order_graph(data, pg)
    path = nx.shortest_path(og, weight='weight', source=tuple(), target=sort_tuple(tuple(data.columns)))
    order = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        variable = int(str(list(set(v) - set(u))[0]))
        order.append(variable)
    order = np.array(order)
    np_data = np_data.T
    return np_data, order


def generator(n=100):
    X_data = []
    Y_data = []
    for i in range(n):
        x, y = dp_generate()
        X_data.append(x)
        Y_data.append(y)
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    print(n, "generated!")
    return X_data, Y_data


def data_augment(x, y, aug_rate=10):
    """

    :param x:
    :param y:
    :return:
    """
    x_list = []
    y_list = []
    num_samples = len(x)
    for i in tqdm(range(num_samples)):
        data = x[i]
        order = y[i]
        x_list.append(data)
        y_list.append(order)
        for j in range(aug_rate - 1):
            np.random.shuffle(data)
            x_list.append(data)
            y_list.append(order)
    new_x = np.array(x_list)
    new_y = np.array(y_list)

    return new_x, new_y


if __name__ == '__main__':
    x = np.load(r"x_5_1000000.npy")
    y = np.load(r"y_5_1000000.npy")
    new_x, new_y = data_augment(x,y)
    np.save(r"x_5_10000000_aug.npy",new_x)
    np.save(r"y_5_10000000_aug.npy", new_y)

