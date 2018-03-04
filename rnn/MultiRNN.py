# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        MultiRNN 
# Date:             2/28/18 8:35 PM 
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


def get_a_cell():
	# 每调用一次这个函数就返回一个BasicRNNCell
	return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell(
		[get_a_cell() for _ in range(3)]
		)  # 3层RNN

# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size)  # (128, 128, 128)

# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32 是 batch_size
h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
output, h1 = cell.call(inputs, h0)
print(h1)  # tuple中含有3个32x128的向量
