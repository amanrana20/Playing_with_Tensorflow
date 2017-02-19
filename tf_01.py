'''
Author: Aman Rana
Topic: Adding two constants / variabes in tensorflow
'''

import tensorflow as tf


## Constants first
with tf.name_scope('Input_A'):
	a = tf.constant(32., name='a')
	tf.summary.scalar('a', a)
	b = tf.constant(64., name='b')
	tf.summary.scalar('b', b)
	sum = a + b
	tf.summary.scalar('sum', sum)

## Adding Variables using placeholders
with tf.name_scope('Input_B'):
	_a = tf.placeholder(tf.float32, name='_A')
	tf.summary.scalar('A', _a)
	_b = tf.placeholder(tf.float32, name='_B')
	tf.summary.scalar('B', _b)
	_sum = _a + _b
	tf.summary.scalar('SUM', _sum)

## Startig a sessiion and running it
sess = tf.Session()
sess.run(tf.global_variables_initializer())
usr_inp_a, usr_inp_b = map(int, raw_input('Enter the two numbers separated by a space: ').split(' '))
with tf.name_scope('Output'):
	total_sum = sum + _sum
print 'Using sess.run(feed_dict=):', sess.run(total_sum, feed_dict={_a: usr_inp_a, _b: usr_inp_b})

## Merging the summaries and 
_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('.', sess.graph)





# creating a graph for sess1
