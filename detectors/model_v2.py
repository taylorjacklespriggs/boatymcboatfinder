import tensorflow as tf

from constants import activation_functions, assignments, x, y, training_mode
from model_base import ModelBase

def conv2d(in_tensor, kernel, output):
    in_features = in_tensor.get_shape()[3].value
    return tf.nn.conv2d(
      inputs=in_tensor,
      filter=[kernel, kernel, in_features, output],
      strides=[1, 1, 1, 1],
      padding='SAME',
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
    )

def bn(in_tensor):
  return tf.layers.batch_normalization(in_tensor, training=training_mode)

class ModelV2(ModelBase):
  def create_model(self):
    relu = activation_functions['relu']
    sigmoid = activation_functions['sigmoid']
    in_tensor = x
    for i in range(assignments.get('num_conv', 2)):
      block_features = assignments.get('block_features', 256)
      shortcut = bn(conv2d(in_tensor, 1, block_features))
      in_tensor = relu(bn(conv2d(
        in_tensor=in_tensor,
        kernel=assignments.get('conv1_kernel', 5),
        output=assignments.get('conv1_features', 256),
      )))
      in_tensor = bn(conv2d(
        in_tensor=in_tensor,
        kernel=assignments.get('conv2_kernel', 5),
        output=block_features,
      ))
      in_tensor = in_tensor + shortcut

    output_1 = relu(bn(conv2d(
      in_tensor=in_tensor,
      kernel=assignments.get('output1_kernel', 5),
      output=assignments.get('output1_features', 256),
    )))
    activation = sigmoid(bn(conv2d(
      in_tensor=in_tensor,
      kernel=assignments.get('activation_kernel', 5),
      output=1,
    )))

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV2)
