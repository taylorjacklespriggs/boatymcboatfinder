import tensorflow as tf

from constants import activation_functions, assignments, x, y, training_mode
from model_base import ModelBase

def create_conv_block(in_tensor, kernel, output, activation):
    in_features = in_tensor.get_shape()[3].value
    w_c = tf.Variable(tf.random_normal([kernel, kernel, in_features, output]))
    b_c = tf.Variable(tf.random_normal([output]))
    conv = tf.nn.conv2d(in_tensor, w_c, strides=[1, 1, 1, 1], padding='SAME')
    conv_w_bias = tf.add(conv, b_c)
    return activation_functions[activation](conv_w_bias)

class ModelV2(ModelBase):
  def __init__(self, n_conv):
    self.n_conv = n_conv
    super().__init__()

  def create_model(self):
    in_tensor = x
    for i in range(self.n_conv):
      in_tensor = create_conv_block(
          in_tensor=in_tensor,
          kernel=assignments.get('conv{}_kernel'.format(i), 5),
          output=assignments.get('conv{}_output'.format(i), 64),
          activation=assignments.get('conv{}_act'.format(i), 'relu'),
      )
      in_tensor = tf.layers.batch_normalization(in_tensor, training=training_mode)

    return create_conv_block(
        in_tensor=in_tensor,
        kernel=assignments.get('output_kernel', 5),
        output=1,
        activation=assignments.get('output_act', 'sigmoid'),
    )
