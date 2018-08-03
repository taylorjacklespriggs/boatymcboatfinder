import tensorflow as tf

from model_v1 import ModelV1
from data_loader import TrainingDataLoader
from constants import assignments, x, y

if __name__ == '__main__':
  loader = TrainingDataLoader(
      'ec2-54-245-11-96.us-west-2.compute.amazonaws.com:5000',
    batch_size=assignments['batch_size'],
    image_size=assignments['image_size'],
  )
  with tf.Session() as sess:
    model = ModelV1()
    sess.run(tf.global_variables_initializer())
    for _ in range(20):
      batch = loader.load_train_batch()
      print(model.train_batch(sess, batch))
