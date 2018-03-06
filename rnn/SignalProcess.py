# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        SignalProcess 
# Date:             3/4/18 8:47 AM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/RNN
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
import tensorflow as tf
import argparse
import os

EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
PREDICT_BATCH_SIZE = 1
PREDICT_SAVED_DIRECTORY = '../data_set/my_set/predictions'
EPS = 10e-5
FLAGS = None
CLASS_NUM = 4
TIME_STEP = 2600
UNITS_NUM = 128
TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
CHECK_POINT_PATH = None


def write_data_to_tfrecords():
	import numpy as np
	import pandas as pd
	type_to_num = {
		'galaxy': 0,
		'qso': 1,
		'star': 2,
		'unknown': 3
	}
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, TRAIN_SET_NAME))
	validation_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME))
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, TEST_SET_NAME))

	train_set = pd.read_csv(filepath_or_buffer=os.path.join(FLAGS.data_dir, 'train_set.csv'), header=0, sep=',')
	# splite_merge_csv()
	# print(train_set.head())
	row_num = train_set.shape[0]
	for index, row in train_set.iterrows():
		# print(row['id'])
		train_list = np.loadtxt(
			os.path.join('../data/first_train_data_20180131', '%d.txt' % row['id']), delimiter=",", skiprows=0, dtype=np.float32)

		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[type_to_num[row['type']]])),
			'signal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_list.tobytes()]))
		}))
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done train_set writing %.2f%%' % (index / row_num * 100))
	train_set_writer.close()
	print("Done train_set writing")

	validation_set = pd.read_csv(filepath_or_buffer=os.path.join(FLAGS.data_dir, 'validation_set.csv'), header=0, sep=',')
	# splite_merge_csv()
	# print(validation_set.head())
	row_num = validation_set.shape[0]
	for index, row in validation_set.iterrows():
		# print(row['type'])
		validation_list = np.loadtxt(
			os.path.join('../data/first_train_data_20180131', '%d.txt' % row['id']), delimiter=",", skiprows=0, dtype=np.float32)

		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[type_to_num[row['type']]])),
			'signal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[validation_list.tobytes()]))
		}))
		validation_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done validation_set writing %.2f%%' % (index / row_num * 100))
	validation_set_writer.close()
	print("Done validation_set writing")

	test_set = pd.read_csv(filepath_or_buffer=os.path.join(FLAGS.data_dir, 'first_test_index_20180131.csv'), header=0, sep=',')
	# splite_merge_csv()
	# print(test_set.head())
	row_num = test_set.shape[0]
	for index, row in test_set.iterrows():
		# print(row['type'])
		test_list = np.loadtxt(
			os.path.join('../data/first_test_data_20180131', '%d.txt' % row['id']), delimiter=",", skiprows=0, dtype=np.float32)

		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['id']])),
			'signal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_list.tobytes()]))
		}))
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done test_set writing %.2f%%' % (index / row_num * 100))
	test_set_writer.close()
	print("Done test_set writing")


def read_image(file_queue):
	reader = tf.TFRecordReader()
	# key, value = reader.read(file_queue)
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'label': tf.FixedLenFeature([], tf.int64),
			'signal': tf.FixedLenFeature([], tf.string)
			})

	signal = tf.decode_raw(features['signal'], tf.float32)
	# print('image ' + str(image))
	# image = tf.reshape(image, [INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL])
	# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	# image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDE))
	# signal = tf.cast(features['signal'], tf.float32)
	signal = tf.reshape(signal, [2600, 1])

	# label = tf.decode_raw(features['label'], tf.int64)
	label = tf.cast(features['label'], tf.int32)
	# label = tf.reshape(label, [OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT])
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])
	return signal, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	min_after_dequeue = 2000
	capacity = 4000
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
	one_hot_labels = tf.reshape(label_batch, [batch_size])
	return image_batch, one_hot_labels


def read_check_tfrecords():
	train_file_path = os.path.join(FLAGS.data_dir, 'train_set.tfrecords')
	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	# train_images, train_labels = read_image(train_image_filename_queue)
	train_images, train_labels = read_image_batch(file_queue=train_image_filename_queue, batch_size=5)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=train_labels, depth=CLASS_NUM))
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		signal, label = sess.run([train_images, train_labels])
		print(signal)
		print(signal.shape)
		print(label)
		# print(sess.run(one_hot_labels))
		coord.request_stop()
		coord.join(threads)
	print("Done reading and checking")


def splite_merge_csv():
	import pandas as pd
	df = pd.read_csv(filepath_or_buffer='../data/first_train_index_20180131.csv', header=0, sep=',')
	train_set = pd.DataFrame()
	validate_set = pd.DataFrame()
	# print(df.head())
	groups = df.groupby('type')
	print(groups.count())
	for name, group in groups:
		if name == 'galaxy':
			train_set = pd.concat([train_set, group[:5200]])
			validate_set = pd.concat([validate_set, group[5200:]])
		elif name == 'qso':
			train_set = pd.concat([train_set, group[:1300]])
			validate_set = pd.concat([validate_set, group[1300:]])
		elif name == 'star':
			train_set = pd.concat([train_set, group[40500:81000]])
			validate_set = pd.concat([validate_set, group[81000:81969]])
		elif name == 'unknown':
			print(name)
			train_set = pd.concat([train_set, group[:34000]])
			validate_set = pd.concat([validate_set, group[34000:]])
	print('train_set')
	print(train_set.count(axis=0))
	print('validate_set')
	print(validate_set.count(axis=0))
	train_set.sample(frac=1).to_csv(path_or_buf='../data/train_set.csv', index=False)
	validate_set.sample(frac=1).to_csv(path_or_buf='../data/validation_set.csv', index=False)
	print('Done splite and merge csv')


class RNN:
	def __init__(self):
		print('new LTSM RNN')
		self.time_step = TIME_STEP
		self.num_units = UNITS_NUM
		self.batch_size = 0
		self.input_signal, self.input_label = [None] * 2
		self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
		self.prediction, self.correct_prediction, self.accuracy = [None] * 3
		self.keep_prob, self.lamb, self.is_traing = [None] * 3
		self.num_class = CLASS_NUM

	@staticmethod
	def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
		from tensorflow.python.training.moving_averages import assign_moving_average

		with tf.variable_scope(name):
			params_shape = x.shape[-1:]
			moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
			moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

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
				normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,  variance_epsilon=eps)
			else:
				normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=eps)
			return normed

	def set_up_network(self, batch_size):

		print('setting up RNN')

		with tf.name_scope('inputs'):
			self.batch_size = batch_size
			self.input_signal = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.time_step, 1])
			self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
			self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
			self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')

			normed_input_signal = self.batch_norm(x=self.input_signal, is_training=self.is_traing, name='input')
			# normed_input_signal = self.input_signal
			# cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
			cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
			h0 = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
			outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=normed_input_signal, initial_state=h0)
			print('final_state shape: ' + str(final_state.shape))  # (batch_size, 128)
			print('outputs shape: ' + str(outputs[:, -1, :].shape))  # (batch_size, time_step, 128)
			# normed_final_state_h = self.batch_norm(x=final_state.h, is_training=self.is_traing, name='output')
			normed_outputs = self.batch_norm(x=outputs[:, -1, :], is_training=self.is_traing, name='output')
			# normed_final_state_h = final_state.h

		with tf.name_scope('fnn'), tf.variable_scope(name_or_scope='rnn'):
			w = tf.get_variable(
				name='w', shape=[self.num_units, self.num_class], dtype=tf.float32,
				initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(x=2 / (self.num_units * self.num_class))))
			# b = tf.get_variable(
			# 	name='b', shape=[self.num_class], dtype=tf.float32,
			# 	initializer=tf.truncated_normal_initializer(stddev=0.1))
			result_fnn = tf.matmul(a=normed_outputs, b=w, name='matmul')
			# result_bias_add = tf.nn.bias_add(value=result_fnn, bias=b, name='bias_add')

		with tf.name_scope('softmax_loss'):
			# self.prediction = tf.nn.bias_add(value=result_fnn, bias=b, name='bias_add')
			# print(prediction .shape)
			normed_prediction = self.batch_norm(x=result_fnn, is_training=self.is_traing, name='prediction')
			# normed_prediction = self.prediction
			self.prediction = normed_prediction
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=self.input_label, logits=self.prediction, name='loss')
			self.loss_mean = tf.reduce_mean(self.loss)

		with tf.name_scope('accuracy'):
			self.correct_prediction = \
				tf.equal(tf.argmax(input=self.prediction, axis=-1, output_type=tf.int32), self.input_label)
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_mean)

		print('Done setting up RNN')

	def train(self, batch_size):
		# set_up RNN
		self.set_up_network(batch_size)
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

		train_file_path = os.path.join(FLAGS.data_dir, TRAIN_SET_NAME)
		train_image_filename_queue = tf.train.string_input_producer(
			string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
		train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
		tf.summary.scalar("loss", self.loss_mean)
		tf.summary.scalar('accuracy', self.accuracy)
		merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			print('Start trianing')
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# print(label)
					lo, acc, summary_str = sess.run(
						[self.loss_mean, self.accuracy, merged_summary],
						feed_dict={
							self.input_signal: example, self.input_label: label, self.keep_prob: 1.0,
							self.lamb: 0.004, self.is_traing: True}
					)
					summary_writer.add_summary(summary_str, epoch)
					# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					if epoch % 1 == 0:
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					sess.run(
						[self.train_step],
						feed_dict={
							self.input_signal: example, self.input_label: label, self.keep_prob: 0.8,
							self.lamb: 0.004, self.is_traing: True}
					)
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				all_parameters_saver.save(sess=sess, save_path=ckpt_path)
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print("Done training")

	def validate(self, batch_size):
		self.set_up_network(batch_size)
		validation_file_path = os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME)
		validation_image_filename_queue = tf.train.string_input_producer(
			string_tensor=tf.train.match_filenames_once(validation_file_path), num_epochs=1, shuffle=True)
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")  # CHECK_POINT_PATH
		validation_images, validation_labels = read_image_batch(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
		# tf.summary.scalar("loss", self.loss_mean)
		# tf.summary.scalar('accuracy', self.accuracy)
		# merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([validation_images, validation_labels])  # 在会话中取出image和label
					# print(label)
					lo, acc = sess.run(
						[self.loss_mean, self.accuracy],
						feed_dict={
							self.input_signal: example, self.input_label: label, self.keep_prob: 1.0,
							self.lamb: 0.004, self.is_traing: True}
					)
					# summary_writer.add_summary(summary_str, epoch)
					# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					if epoch % 1 == 0:
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done validating -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print('Done validating')

	def predict(self):
		# import numpy as np
		import pandas as pd
		num_to_type = {
			0: 'galaxy',
			1: 'qso',
			2: 'star',
			3: 'unknown'
		}
		id_to_type = {}
		self.set_up_network(batch_size=TEST_BATCH_SIZE)
		test_file_path = os.path.join(FLAGS.data_dir, TEST_SET_NAME)
		test_set = pd.read_csv(filepath_or_buffer=os.path.join(FLAGS.data_dir, 'test_set_test.csv'), header=0, sep=',')
		test_image_filename_queue = tf.train.string_input_producer(
			string_tensor=tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=True)
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")  # CHECK_POINT_PATH
		test_images, test_labels = read_image_batch(test_image_filename_queue, TEST_BATCH_SIZE)
		row_num = test_set.shape[0]
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([test_images, test_labels])  # 在会话中取出image和label
					# print(label)
					predict_num = sess.run(
						tf.argmax(input=self.prediction, axis=-1),
						feed_dict={
							self.input_signal: example, self.keep_prob: 1.0,
							self.lamb: 0.004, self.is_traing: True}
					)
					# summary_writer.add_summary(summary_str, epoch)
					# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					for index, the_id in enumerate(label):
						id_to_type[the_id] = num_to_type[predict_num[index]]
					print('Done testing %.2f%%' % (TEST_BATCH_SIZE * epoch / row_num * 100))
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done testing -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		test_set['prediction class'] = test_set['id'].map(id_to_type)
		test_set.to_csv(path_or_buf='../data/test_set_test.csv', index=False)
		print('Done prediction')


def main():
	# splite_merge_csv()
	# write_data_to_tfrecords()
	# test()
	# read_check_tfrecords()
	rnn = RNN()
	# rnn.set_up_network(batch_size=TRAIN_BATCH_SIZE)
	# rnn.train(batch_size=TRAIN_BATCH_SIZE)
	# rnn.validate(batch_size=256)
	# rnn.predict()
	# read_check_tfrecords()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 数据地址
	parser.add_argument(
		'--data_dir', type=str, default='../data',
		help='Directory for storing input data')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='../data/saved_models',
		help='output model path')

	# 日志保存地址
	parser.add_argument(
		'--tb_dir', type=str, default='../data/logs',
		help='TensorBoard log path')

	FLAGS, _ = parser.parse_known_args()
	main()
