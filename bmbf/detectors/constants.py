from keras import backend as K
from keras.layers import Concat, Input
from keras.optimizers import Adam
import tensorflow as tf

try:
  import galileo.io
  assignments = galileo.io.suggestion.assignments.copy()
except ImportError:
  assignments = {}

full_size = 768

batch_size = tf.placeholder(tf.int32)
input = Concat(axis=3)([
  Input(shape=(full_size, full_size, 3,)),
  K.ones(shape=(full_size, full_size, 1)),
])

optimizer = Adam(lr=assignments.setdefault('log_learning_rate', -3))
