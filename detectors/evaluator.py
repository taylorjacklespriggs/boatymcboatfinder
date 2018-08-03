import math
import tensorflow as tf
import time

from model_v1 import ModelV1
from data_loader import TrainingDataLoader
from constants import assignments, x, y

import galileo_io

evaluations = 200

if __name__ == '__main__':
  training_loader = TrainingDataLoader(
      'ec2-54-245-11-96.us-west-2.compute.amazonaws.com:5000',
    batch_size=assignments['batch_size'],
    image_size=assignments['image_size'],
  )
  evaluation_loader = TrainingDataLoader(
      'ec2-54-245-11-96.us-west-2.compute.amazonaws.com:5000',
    batch_size=1,
    image_size=768,
  )
  with tf.Session() as sess:
    model = ModelV1()
    sess.run(tf.global_variables_initializer())
    start_train = time.time()
    for _ in range(assignments['batches']):
      batch = training_loader.load_train_batch()
      print(model.train_batch(sess, batch))
    galileo_io.log_metric('negative_train_time', start_train - time.time())
    start_eval = time.time()
    loss = sum(
      model.evaluate(sess, evaluation_loader.load_train_batch())
      for _ in range(evaluations)
    ) / evaluations
    galileo_io.log_metric('negative_eval_time', start_eval - time.time())
    print('final loss', loss)
    galileo_io.log_metric('negative_log_loss', -math.log(loss))
