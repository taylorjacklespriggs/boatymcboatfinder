import tensorflow as tf

from constants import activation_functions, assignments, x, y, training_mode
from model_base import ModelBase

def create_conv_block(in_tensor, kernel, output, activation):
    in_features = in_tensor.get_shape()[3].value
    total_in_size = kernel**2 * in_features
    stddev = (2. / total_in_size)**.5
    w_c = tf.Variable(tf.random_normal(
      [kernel, kernel, in_features, output],
      stddev=stddev,
    ))
    conv = tf.nn.conv2d(in_tensor, w_c, strides=[1, 1, 1, 1], padding='SAME')
    bn_conv = tf.layers.batch_normalization(conv, training=training_mode)
    return activation_functions[activation](conv_w_bias)

class ModelV2(ModelBase):
  def create_model(self):
    features = assignments.get('features', 256)
    in_tensor = x
    in_tensor = create_conv_block(
      in_tensor=in_tensor,
      kernel=assignments.get('input_kernel', 7)
      output=features,
      activation='relu',
    )
    in_tensor = tf.layers.batch_normalization(in_tensor, training=training_mode)
    for i in range(assignments.get('num_conv', 2)):
      in_tensor = create_conv_block(
        in_tensor=in_tensor,
        kernel=assignments.get('conv_kernel', 5),
        output=features,
        activation='relu',
      ) + in_tensor

    return create_conv_block(
      in_tensor=in_tensor,
      kernel=assignments.get('output_kernel', 32),
      output=1,
      activation='sigmoid',
    )

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV2)
