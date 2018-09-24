import tensorflow as tf

from bmbf.detectors.constants import activation_functions, assignments, batch_size, x, y, training_mode
from bmbf.detectors.models.model_base import ModelBase

def bn(in_tensor):
  return tf.layers.batch_normalization(in_tensor, training=training_mode)

def conv2d(in_tensor, kernel, out_features, activation='relu'):
  in_features = in_tensor.shape[3].value
  w_c = tf.Variable(tf.random_normal(
    [kernel, kernel, in_features, out_features],
    dtype=tf.float32,
  ) * tf.sqrt(2 / in_features))
  conv = tf.nn.conv2d(in_tensor, w_c, strides=[1, 1, 1, 1], padding='SAME')
  return activation_functions[activation](conv)

def max_pool(in_tensor, size):
  return tf.nn.max_pool(
    in_tensor,
    ksize=[1, size, size, 1],
    strides=[1, size, size, 1],
    padding='SAME',
  )

def up_conv2d(in_tensor, kernel, result_size):
  in_shape = in_tensor.shape
  in_features = in_shape[3].value
  w_c = tf.Variable(tf.random_normal(
    [kernel, kernel, in_features, in_features],
    dtype=tf.float32,
  ) * tf.sqrt(2 / in_features))
  conv = tf.nn.conv2d_transpose(
    in_tensor,
    w_c,
    [batch_size, result_size, result_size, in_features],
    strides=[1, kernel, kernel, 1],
    padding='SAME',
  )
  return conv

def n_conv_block(in_tensor, num_convolutions, kernel, out_features):
  if num_convolutions:
    return bn(n_conv_block(
      conv2d(in_tensor, kernel, out_features),
      num_convolutions - 1,
      kernel,
      out_features,
    ))
  else:
    return in_tensor

def skip_network(in_tensor, layer_params, base_params, size=768):
  if layer_params:
    (
      in_num_convs,
      in_conv_kernel,
      in_conv_features,
      pool_size,
      out_num_convs,
      out_conv_kernel,
      out_conv_features,
    ), *rest = layer_params
    in_tensor = n_conv_block(in_tensor, in_num_convs, in_conv_kernel, in_conv_features)
    skip_connection = in_tensor
    in_tensor = max_pool(in_tensor, pool_size)
    in_tensor = skip_network(in_tensor, rest, base_params, size=size//pool_size)
    in_tensor = bn(up_conv2d(in_tensor, pool_size, size))
    in_tensor = tf.concat([in_tensor, skip_connection], 3)
    in_tensor = n_conv_block(in_tensor, out_num_convs, out_conv_kernel, out_conv_features)
    return in_tensor
  else:
    return n_conv_block(in_tensor, *base_params)

N_CONV_NUM = 2
N_CONV_KERNEL = 3
POOL_SIZE = 2
BASE_POWER = 3
DEPTH = 1
class M1(ModelBase):
  def create_model(self):
    in_tensor = x
    skip_net = skip_network(
      in_tensor,
      [(
        N_CONV_NUM,
        N_CONV_KERNEL,
        2**(i + BASE_POWER),
        POOL_SIZE,
        N_CONV_NUM,
        N_CONV_KERNEL,
        2**(i + BASE_POWER),
      ) for i in range(DEPTH)],
      (N_CONV_NUM, N_CONV_KERNEL, 2**(DEPTH + BASE_POWER)),
    )
    return bn(tf.sigmoid(conv2d(skip_net, 1, 1)))


if __name__ == '__main__':
  from bmbf.detectors.evaluator import train_and_evaluate
  train_and_evaluate(M1)
