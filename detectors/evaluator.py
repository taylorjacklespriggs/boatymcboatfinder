import math
import tensorflow as tf
import time

from data_loader import TrainingDataLoader, load_evaluation_data
from constants import assignments, x, y

def evaluate_model(session, model, evaluation_data):
  count = 10
  x_eval, y_eval = evaluation_data
  n = 0
  blank_total = 0.
  boats_total = 0.
  for i in range(0, len(x_eval), count):
      end = min(i + count, len(x_eval))
      spliced = x_eval[i:end], y_eval[i:end]
      blank, boat = model.evaluate(session, spliced)
      blank_total += blank
      boats_total += boat
      n += 1
      print('average blank', blank_total / n, 'boats', boats_total / n)
  return blank_total / n, boats_total / n

def train_and_evaluate(model_gen):
  import galileo_io

  server = 'ec2-54-245-11-96.us-west-2.compute.amazonaws.com:5000'
  training_loader = TrainingDataLoader(
    server,
    batch_size=assignments['batch_size'],
    image_size=assignments['image_size'],
    blank_prob=assignments['blank_prob'],
  )
  evaluation_data = load_evaluation_data(server)
  with tf.Session() as sess:
    model = model_gen()
    sess.run(tf.global_variables_initializer())
    start_train = time.time()
    load_time = 0.
    batch_time = 0.
    batches = assignments['batches']
    for _ in range(batches):
      start_load = time.time()
      batch = training_loader.load_batch()
      load_time += time.time() - start_load
      start_batch = time.time()
      print(model.train_batch(sess, batch))
      batch_time += time.time() - start_batch
    galileo_io.log_metric('negative_train_time', start_train - time.time())
    galileo_io.log_metadata('average_load_time', load_time / batches)
    galileo_io.log_metadata('average_batch_time', batch_time / batches)
    start_eval = time.time()
    blank_loss, boat_loss = evaluate_model(sess, model, evaluation_data)
    galileo_io.log_metric('negative_eval_time', start_eval - time.time())
    print('final blank', blank_loss, 'boats', boat_loss)
    galileo_io.log_metadata('blank_loss', blank_loss)
    galileo_io.log_metadata('boat_loss', boat_loss)
    galileo_io.log_metric('negative_log_loss', -math.log(blank_loss + boat_loss))
