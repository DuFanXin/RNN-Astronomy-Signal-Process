# -*- coding:utf-8 -*-
"""
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        BasicRNN
# Date:             2/26/18 8:15 PM 
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

BATCH_SIZE = 2
inp = np.zeros(shape=[BATCH_SIZE, 10, 1], dtype=np.float32)
inp[0, :, 0] = [i for i in range(10)]
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)  # state_size = 128
print(cell.state_size)  # 128

inputs = tf.placeholder(dtype=np.float32, shape=[BATCH_SIZE, 10, 1])  # 32 是 batch_size
h0 = cell.zero_state(batch_size=BATCH_SIZE, dtype=np.float32)  # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
# inpu = tf.reshape(tensor=inputs[:, 0], shape=[BATCH_SIZE, 1])
# output, h1 = cell.call(inputs=inpu, state=h0)  # 调用call函数
h = h0
# print(input[0].shape)
with tf.variable_scope(name_or_scope='rnn'):
	# tf.get_variable_scope().reuse_variables()
	for i in range(10):
		# inpu = tf.reshape(tensor=inputs[:, i], shape=[BATCH_SIZE, 1])
		output, h1 = cell(inputs=inputs[:, i, :], state=h)  # 调用call函数
		h = h1
	print(h.shape)  # (32, 128)
# with tf.Session() as sess:  # 开始一个会话
# 	sess.run(tf.global_variables_initializer())
# 	sess.run(tf.local_variables_initializer())
# 	inpu = sess.run(inputs, feed_dict={inputs: inp})
# 	print(inpu[0][1])
