# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RNN-SignalProcess
# File Name:        TestSomeFunction 
# Date:             3/1/18 7:45 PM 
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


def write_img_to_tfrecords():
	import numpy as np
	import pandas as pd
	type_to_num = {
		'galaxy': 0,
		'qso': 1,
		'star': 2,
		'unknown': 3
	}
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))
	validation_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'validation_set.tfrecords'))

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


def test():
	import numpy as np
	# import pandas as pd
	type_to_num = {
		'galaxy':   0,
		'qso':      1,
		'star':     2,
		'unknown':  3
	}
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set_test.tfrecords'))  # 要生成的文件
	train_list = np.loadtxt(os.path.join('../data', '%d.txt' % 696220), delimiter=",", skiprows=0, dtype=np.float32)

	example = tf.train.Example(features=tf.train.Features(feature={
		'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[type_to_num['star']])),
		'signal': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_list.tobytes()]))
	}))
	train_set_writer.write(example.SerializeToString())  # 序列化为字符串

	train_set_writer.close()


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


def read_check_tfrecords():
	train_file_path = os.path.join(FLAGS.data_dir, 'train_set.tfrecords')
	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	train_images, train_labels = read_image(train_image_filename_queue)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=train_labels, depth=CLASS_NUM))
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		signal, label = sess.run([train_images, train_labels])
		print(signal)
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
	grouped = df.groupby('type')
	print(grouped.count())
	for name, group in grouped:
		if name == 'galaxy':
			train_set = pd.concat([train_set, group[:5200]])
			validate_set = pd.concat([validate_set, group[5200:]])
		elif name == 'qso':
			train_set = pd.concat([train_set, group[:1300]])
			validate_set = pd.concat([validate_set, group[1300:]])
		elif name == 'star':
			train_set = pd.concat([train_set, group[:140000]])
			validate_set = pd.concat([validate_set, group[140000:140969]])
		elif name == 'unknown':
			print(name)
			train_set = pd.concat([train_set, group[:34000]])
			validate_set = pd.concat([validate_set, group[34000:]])
	print('train_set')
	print(train_set.count(axis=0))
	print('validate_set')
	print(validate_set.count(axis=0))
	train_set.sample(frac=1).to_csv(path_or_buf='../data/train_set.csv')
	validate_set.sample(frac=1).to_csv(path_or_buf='../data/validation_set.csv')
	print('Done splite and merge csv')


def main():
	# splite_merge_csv()
	# write_img_to_tfrecords()
	# test()
	read_check_tfrecords()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 数据地址
	parser.add_argument(
		'--data_dir', type=str, default='../data',
		help='Directory for storing input data')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='../data_set/saved_models',
		help='output model path')

	# 日志保存地址
	parser.add_argument(
		'--tb_dir', type=str, default='../data_set/logs',
		help='TensorBoard log path')

	FLAGS, _ = parser.parse_known_args()
	main()
