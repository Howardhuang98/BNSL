#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_dataset.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/4/30 11:11  
------------      
"""
import unittest

from bnsl.dataset import Dataset

class Test_genetic(unittest.TestCase):

    def test_asian(self):
        self.asian = Dataset('asian')
        print(self.asian)

    def test_insurance(self):
        self.insurance = Dataset('insurance')
        print(self.insurance)