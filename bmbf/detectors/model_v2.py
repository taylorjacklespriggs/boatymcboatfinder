import tensorflow as tf

from constants import activation_functions, assignments, x, y, training_mode
from model_base import ModelBase

def conv2d(in_tensor, kernel, output):
  return tf.layers.conv2d(
    inputs=in_tensor,
    filters=output,
    kernel_size=kernel,
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
    for i in range(assignments.get('num_conv', 1)):
      block_features = assignments.get('block_features', 1)
      shortcut = bn(conv2d(in_tensor, 1, block_features))
      in_tensor = relu(bn(conv2d(
        in_tensor=in_tensor,
        kernel=assignments.get('conv1_kernel', 1),
        output=assignments.get('conv1_features', 1),
      )))
      in_tensor = bn(conv2d(
        in_tensor=in_tensor,
        kernel=assignments.get('conv2_kernel', 1),
        output=block_features,
      ))
      in_tensor = in_tensor + shortcut

    output_1 = relu(bn(conv2d(
      in_tensor=in_tensor,
      kernel=assignments.get('output1_kernel', 1),
      output=assignments.get('output1_features', 1),
    )))
    activation = sigmoid(bn(conv2d(
      in_tensor=output_1,
      kernel=assignments.get('activation_kernel', 1),
      output=1,
    )))
    return activation

if __name__ == '__main__':
  from evaluator import train_and_evaluate
  train_and_evaluate(ModelV2)
