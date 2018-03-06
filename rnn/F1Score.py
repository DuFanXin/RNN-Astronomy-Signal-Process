# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-Astronomy-Signal-Process 
# File Name:        F1Score 
# Date:             3/5/18 5:21 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/RNN-Astronomy-Signal-Process
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
file_name = '../data/test_set1.csv'


def split_true_or_false_label():
	df = pd.read_csv(filepath_or_buffer=file_name, header=0, sep=',')

	df['galaxy true'] = df['type'].map(lambda x: 1 if (x == 'galaxy') else 0)
	df['galaxy predict'] = df['prediction'].map(lambda x: 1 if (x == 'galaxy') else 0)

	df['qso true'] = df['type'].map(lambda x: 1 if (x == 'qso') else 0)
	df['qso predict'] = df['prediction'].map(lambda x: 1 if (x == 'qso') else 0)

	df['star true'] = df['type'].map(lambda x: 1 if (x == 'star') else 0)
	df['star predict'] = df['prediction'].map(lambda x: 1 if (x == 'star') else 0)

	df['unknown true'] = df['type'].map(lambda x: 1 if (x == 'unknown') else 0)
	df['unknown predict'] = df['prediction'].map(lambda x: 1 if (x == 'unknown') else 0)

	df.to_csv(path_or_buf=file_name, index=False)


def score(label_type):
	df = pd.read_csv(filepath_or_buffer=file_name, header=0, sep=',')
	y_true = df[label_type + ' true'].tolist()
	y_pred = df[label_type + ' predict'].tolist()

	# y_true = np.reshape(y_true, [-1])
	# y_pred = np.reshape(y_pred, [-1])
	# print(df['galaxy predict'].tolist())
	p = precision_score(y_true, y_pred, average='binary')
	r = recall_score(y_true, y_pred, average='binary')
	f1score = f1_score(y_true, y_pred, average='binary')

	print(p)
	print(r)
	print(f1score)

# split_true_or_false_label()
score(label_type='star')
