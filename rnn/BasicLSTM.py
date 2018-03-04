# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        BasicLSTM 
# Date:             2/28/18 8:33 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/RNN
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
import tensorflow as tf
import numpy as np
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
