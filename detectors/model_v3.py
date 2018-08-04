import tensorflow as tf

from constants import activation_functions, assignments, x, y, training_mode
from model_base import ModelBase

def create_conv(in_tensor, kernel, out_features, activation='relu'):
  in_features = in_tensor.get_shape()[3].value
  w_c = tf.Variable(tf.random_normal([kernel, kernel, in_features, out_features]))
  b_c = tf.Variable(tf.random_normal([out_features]))
  conv = tf.nn.conv2d(in_tensor, w_c, strides=[1, 1, 1, 1], padding='SAME')
  conv_w_bias = tf.add(conv, b_c)
  return activation_functions[activation](conv_w_bias)

def create_inception_block(in_tensor):
    in_features = in_tensor.get_shape()[3].value

    # 'conv_a' is a single 1x1 convolution
    conv_a1 = create_conv(in_tensor, 1, assignments.get('conv_a1_features', 16))
    conv_a = conv_a1

    # 'conv_b' is a 1x1 convolution followed by 3x3 convolution
    conv_b1 = create_conv(in_tensor, 1, assignments.get('conv_b1_features', 16))
    conv_b2 = create_conv(conv_b1, 3, assignments.get('conv_b2_features', 16))
    conv_b = conv_b2

    # 'conv_c' is a 1x1 convolution followed by 5x5 convolution
    conv_c1 = create_conv(in_tensor, 1, assignments.get('conv_c1_features', 16))
    conv_c2 = create_conv(conv_c1, 5, assignments.get('conv_c2_features', 16))
    conv_c = conv_c2

    # 'conv_d' is a 3x3 max pool followed by 1x1 convolution
    conv_d1 = tf.nn.max_pool(
      in_tensor,
      ksize=[1, 3, 3, 1],
      strides=[1, 1, 1, 1],
      padding='SAME',
    )
    conv_d2 = create_conv(conv_c1, 1, assignments.get('conv_d2_features', 16))
    conv_d = conv_d2

    return activation_functions[assignments.get('block_act', 'relu')](
      tf.concat([conv_a, conv_b, conv_c, conv_d], 3)
    )

class ModelV3(ModelBase):
  def __init__(self, n_conv):
    self.n_conv = n_conv
    super().__init__()

  def create_model(self):
    in_tensor = x
    for i in range(assignments['num_conv']):
      in_tensor = create_inception_block(in_tensor=in_tensor)
      in_tensor = tf.layers.batch_normalization(in_tensor, training=training_mode)

    return create_conv(
        in_tensor=in_tensor,
        kernel=assignments.get('output_kernel', 5),
        out_features=1,
        activation=assignments.get('output_act', 'sigmoid'),
    )

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV3)
