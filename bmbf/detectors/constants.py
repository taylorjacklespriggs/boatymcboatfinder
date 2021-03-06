import tensorflow as tf

try:
  import galileo.io
  assignments = galileo.io.suggestion.assignments.copy()
except ImportError:
  assignments = {}

full_size = 768

batch_size = tf.placeholder(tf.int32)
x = tf.placeholder(tf.float32, [None, None, None, 4])
y = tf.placeholder(tf.float32, [None, None, None, 1])
training_mode = tf.placeholder("bool")
learning_rate = tf.placeholder(tf.float32)

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

optimizer = optimizers[assignments.setdefault('optimizer', 'adam')]
