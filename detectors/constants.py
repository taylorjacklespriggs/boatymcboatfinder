import tensorflow as tf

try:
  import galileo.io
  assignments = galileo.io.suggestion.assignments.copy()
except ImportError:
  assignments = {}

image_size = assignments.setdefault('image_size', 768)

x = tf.placeholder("float", [None, None, None, 3])
y = tf.placeholder("float", [None, None, None, 1])
training_mode = tf.placeholder("bool")

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
