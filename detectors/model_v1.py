import tensorflow as tf

from constants import activation_functions, assignments, x, y
from model_base import ModelBase

def create_weights(kernel, in_size, out_size):
  total_in_size = kernel**2 * in_size
  stddev = (2. / total_in_size)**.5
  w = tf.Variable(tf.random_normal([kernel, kernel, in_size, out_size], stddev=stddev))
  b = tf.Variable(tf.zeros([out_size]))
  return w, b

class ModelV1(ModelBase):
  def create_model(self):
    conv1_kernel = assignments.get('conv1_kernel', 5)
    conv1_output = assignments.get('conv1_output', 64)
    w_c1, b_c1 = create_weights(conv1_kernel, 4, conv1_output)
    conv1 = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.add(conv1, b_c1)
    conv1 = activation_functions[assignments.get('conv1_act', 'relu')](conv1)

    conv2_kernel = assignments.get('conv2_kernel', 5)
    w_c2, b_c2 = create_weights(conv2_kernel, conv1_output, 1)
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.add(conv2, b_c2)
    conv2 = activation_functions[assignments.get('conv2_act', 'sigmoid')](conv2)

    return conv2

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV1)
