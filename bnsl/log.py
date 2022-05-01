#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   log.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2022/5/1 14:13  
------------      
"""
import logging
import os
from logging.handlers import RotatingFileHandler

os.makedirs('.logs', exist_ok=True)
bnsl_log = logging.getLogger('bnsl')
bnsl_log.setLevel(logging.INFO)

fh = RotatingFileHandler(os.path.join('.logs', 'log'), maxBytes=5 * 1024 * 1024, backupCount=10, encoding='utf-8')

f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(f)
bnsl_log.addHandler(fh)
