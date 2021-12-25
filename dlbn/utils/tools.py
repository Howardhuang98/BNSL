#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tools.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/30 22:39  
------------      
"""
from multiprocessing import Pool


from generators import *

def generator_multi(n=100,num_of_workers=4):
    if n % num_of_workers !=0:
        raise ValueError("error")
    with Pool(processes=num_of_workers) as pool:
        assign = [1/num_of_workers for i in range(num_of_workers)]
        assign_num = [int(n*i) for i in assign]
        f = generator
        results = pool.map(f, assign_num)
        # obtain 'num_of_workers' result
        X_data = results[0][0]
        Y_data = results[0][1]
        for step, result in enumerate(results):
            if step:
                X_data = np.append(X_data, result[0], axis=0)
                Y_data = np.append(Y_data, result[1], axis=0)
        return X_data, Y_data

if __name__ == '__main__':
    X, Y = generator_multi(1000000, num_of_workers=25)
    np.save(r"x_6_1000000.npy", X)
    np.save(r"y_6_1000000.npy", Y)
