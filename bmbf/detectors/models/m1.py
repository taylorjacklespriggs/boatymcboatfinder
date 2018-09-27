from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, Relu, Sigmoid
import tensorflow as tf

from bmbf.detectors.constants import activation_functions, assignments, batch_size, x, y, training_mode
from bmbf.detectors.models.model_base import ModelBase

def bn(in_tensor, axis=3):
  return BatchNormalization(axis=axis)(in_tensor)

def conv2d(in_tensor, kernel, out_features):
  return Conv2D(
    filters=out_features,
    kernel_size=kernel,
    padding='same',
    use_bias=False,
  )(in_tensor)

def max_pool(in_tensor, size):
  return MaxPooling2D(pool_size=size)(in_tensor)

def up_conv2d(in_tensor, kernel, out_features):
  return Conv2DTranspose(
    filters=out_features,
    kernel_size=kernel,
    use_bias=False,
  )(in_tensor)

def n_conv_block(in_tensor, num_convolutions, kernel, out_features):
  if num_convolutions:
    return n_conv_block(
      Relu(bn(conv2d(in_tensor, kernel, out_features))),
      num_convolutions - 1,
      kernel,
      out_features,
    )
  else:
    return in_tensor

def skip_network(in_tensor, layer_params, base_params):
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
    in_tensor, skip_features = skip_network(in_tensor, rest, base_params)
    in_tensor = bn(up_conv2d(in_tensor, pool_size, skip_features))
    in_tensor = Concat(axis=3)([in_tensor, skip_connection])
    in_tensor = n_conv_block(in_tensor, out_num_convs, out_conv_kernel, out_conv_features)
    return in_tensor, out_conv_features
  else:
    return n_conv_block(in_tensor, *base_params), base_params[3]

N_CONV_NUM = 2
N_CONV_KERNEL = 3
POOL_SIZE = 2
BASE_POWER = 5
DEPTH = 5
class M1(ModelBase):
  def create_model(self):
    in_tensor = x
    skip_net, _ = skip_network(
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
    return Sigmoid(bn(conv2d(skip_net, 1, 1)))


if __name__ == '__main__':
  from bmbf.detectors.evaluator import train_and_evaluate
  train_and_evaluate(M1)
