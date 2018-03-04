# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        DynamicBasicRNN
# Date:             3/1/18 1:03 PM 
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
TIME_STEP = 10
EPS = 1E-5
arr = np.zeros(shape=[BATCH_SIZE, TIME_STEP, 1], dtype=np.float32)
arr[0, :, 0] = [i for i in range(TIME_STEP)]
arr[1, :, 0] = [i for i in range(0, TIME_STEP * 2, 2)]
print(arr)

label = np.zeros(shape=[BATCH_SIZE], dtype=np.int32)
label[0] = 0
label[1] = 1

inputs = tf.placeholder(dtype=np.float32, shape=[BATCH_SIZE, TIME_STEP, 1])  # 32 是 batch_size
input_label = tf.placeholder(dtype=np.int32, shape=[BATCH_SIZE])  # 32 是 batch_size

# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)  # state_size = 128
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# print(cell.state_size)  # 128

# 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
h0 = cell.zero_state(batch_size=BATCH_SIZE, dtype=np.float32)


def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
	from tensorflow.python.training.moving_averages import assign_moving_average

	with tf.variable_scope(name):
		params_shape = x.shape[-1:]
		moving_mean = tf.get_variable(
			name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False
		)
		moving_var = tf.get_variable(
			name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False
		)

		def mean_var_with_update():
			mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
			with tf.control_dependencies([
				assign_moving_average(moving_mean, mean_this_batch, decay),
				assign_moving_average(moving_var, variance_this_batch, decay)
			]):
				return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

		mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
		if affine:  # 如果要用beta和gamma进行放缩
			beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
			gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
			normed = tf.nn.batch_normalization(
				x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=eps
			)
		else:
			normed = tf.nn.batch_normalization(
				x, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=eps
			)
		return normed

with tf.variable_scope(name_or_scope='rnn'):
	w = tf.get_variable(
		shape=[128, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1), name='w'
	)
	b = tf.get_variable(
		shape=[2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1), name='b'
	)
	outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=h0)
	mul = tf.matmul(a=final_state.h, b=w, name='matmul')
	prediction = tf.nn.bias_add(value=mul, bias=b, name='bias_add')
	print('final_state shape: ' + str(final_state.h.shape))   # (2, 128)
	print('outputs shape: ' + str(outputs.shape))    # (2, 10, 128)
# 	# tf.get_variable_scope().reuse_variables()
# 	print(prediction .shape)  # (32, 128)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_label, logits=prediction, name='loss')
	loss_mean = tf.reduce_mean(loss)
	correct_prediction = \
		tf.equal(tf.argmax(input=prediction, axis=-1, output_type=tf.int32), input_label)
	correct_prediction = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)
	train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_mean)

with tf.Session() as sess:  # 开始一个会话
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	for i in range(10):
		acc, loss = sess.run([accuracy, loss_mean], feed_dict={inputs: arr, input_label: label})
		print('acc %.2f' % acc)
		print('loss %.2f' % loss)
		sess.run(train_step, feed_dict={inputs: arr, input_label: label})

