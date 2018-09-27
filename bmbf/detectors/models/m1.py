from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D
import orchestrate.io as orch

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
    strides=kernel,
    use_bias=False,
  )(in_tensor)

def n_conv_block(in_tensor, num_convolutions, kernel, out_features):
  if num_convolutions:
    return n_conv_block(
      Activation('relu')(bn(conv2d(in_tensor, kernel, out_features))),
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
    in_tensor = Concatenate(axis=3)([in_tensor, skip_connection])
    in_tensor = n_conv_block(in_tensor, out_num_convs, out_conv_kernel, out_conv_features)
    return in_tensor, out_conv_features
  else:
    num_convs, kernel, out_features = base_params
    return n_conv_block(in_tensor, num_convs, kernel, out_features), out_features

N_CONV_NUM = orch.assignment('n_conv_num', 2)
N_CONV_KERNEL = orch.assignment('n_conv_kernel', 3)
POOL_SIZE = orch.assignment('pool_size', 2)
BASE_POWER = orch.assignment('base_power', 4)
DEPTH = orch.assignment('depth', 5)
class M1(ModelBase):
  def create_model(self):
    in_tensor = self.input
    skip_net, _ = skip_network(
      in_tensor,
      [(
        N_CONV_NUM,
        N_CONV_KERNEL,
        int(2**(i + BASE_POWER)),
        POOL_SIZE,
        N_CONV_NUM,
        N_CONV_KERNEL,
        int(2**(i + BASE_POWER)),
      ) for i in range(DEPTH)],
      (N_CONV_NUM, N_CONV_KERNEL, int(2**(DEPTH + BASE_POWER))),
    )
    return Activation(orch.assignment('activation', 'relu'))(bn(conv2d(skip_net, 1, 1)))


if __name__ == '__main__':
  from bmbf.detectors.evaluator import train_and_evaluate
  train_and_evaluate(M1)
