import tensorflow as tf

import galileo_io

assignments = galileo_io.suggestion.assignments.copy()

image_size = assignments['image_size']

x = tf.placeholder("float", [None, None, None, 3])
y = tf.placeholder("float", [None, None, None, 1])

activation_functions = {
  'relu': tf.nn.relu,
  'sigmoid': tf.sigmoid,
  'tanh': tf.tanh,
}

optimizers = {
  'gradient_descent': tf.train.GradientDescentOptimizer,
  'rmsprop': tf.train.RMSPropOptimizer,
  'adam': tf.train.AdamOptimizer,
}

optimizer = optimizers[assignments['optimizer']]
