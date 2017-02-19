'''
Author: Aman Rana
Topic: Multi-layer Perceptron and Visualiation using TensorBoard
'''

import tensorflow as tf
import numpy as np


## Generating dataset
with tf.device('/cpu:0'):
	inp_x = np.array([[np.random.randint(0., 1000.) for _ in range(4)] for _ in range(10000)])
	y_hat = np.sum(inp_x, axis=1)
	print inp_x[0]
	test_X = np.array([[np.random.randint(0., 1000.) for _ in range(4)] for _ in range(10000)])
	print test_X[0]
	test_Y = np.sum(test_X, axis=1)
	# print test_Y.shape

## Creating the multi-layer perceptron model
with tf.device('/gpu:0'):
	with tf.name_scope('Parameters'):

		W = {
				1: tf.Variable(tf.truncated_normal([4, 2])), 
				2: tf.Variable(tf.truncated_normal([2, 1]))
			}
		
		B = {
				1: tf.Variable(tf.truncated_normal([10000, 2])),
				2: tf.Variable(tf.truncated_normal([10000, 1]))
			}

def pred(x):
	print 'HERE'
	with tf.device('/gpu:0'):
		
		with tf.name_scope('hidden1'):
			h1 = tf.add(tf.matmul(x, W[1]), B[1])
			h1 = tf.nn.relu(h1)
			# tf.summary.scalar('hidden 1', h1)

		with tf.name_scope('output'):
			out = tf.add(tf.matmul(h1, W[2]), B[2])
			# tf.summary.scalar('out', out)

	return out


with tf.Session() as sess:
	with tf.name_scope('Inputs'):
		X = tf.placeholder(tf.float32, name='X')

	with tf.name_scope('Ground_Truth'):
		Y = tf.placeholder(tf.float32, name='Y')

	y = pred(X)

	## Training
	loss = tf.reduce_sum(tf.square(Y - y))
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)
		
	sess.run(tf.global_variables_initializer())
	sess.run(train, feed_dict={X: inp_x, Y: y_hat})

	_y = pred(X)

	## Testing
	correct_pred = tf.equal(_y, Y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	print '\n\nAccuracy: {}\n\n'.format(sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
	

# train()