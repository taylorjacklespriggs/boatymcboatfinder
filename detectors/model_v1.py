import tensorflow as tf

from constants import activation_functions, assignments, x, y
from model_base import ModelBase

class ModelV1(ModelBase):
  def create_model(self):
    conv1_kernel = assignments.get('conv1_kernel', 5)
    conv1_output = assignments.get('conv1_output', 64)
    w_c1 = tf.Variable(tf.random_normal([conv1_kernel, conv1_kernel, 3, conv1_output]))
    b_c1 = tf.Variable(tf.random_normal([conv1_output]))
    conv1 = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.add(conv1, b_c1)
    conv1 = activation_functions[assignments.get('conv1_act', 'relu')](conv1)

    conv2_kernel = assignments.get('conv2_kernel', 5)
    conv2_output = 1
    w_c2 = tf.Variable(tf.random_normal([conv2_kernel, conv2_kernel, conv1_output, conv2_output]))
    b_c2 = tf.Variable(tf.random_normal([conv2_output]))
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.add(conv2, b_c2)
    conv2 = activation_functions[assignments.get('conv2_act', 'sigmoid')](conv2)

    return conv2

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV1)
