import math
import tensorflow as tf
import time

from data_loader import TrainingDataLoader, load_evaluation_data
from constants import assignments, x, y

def evaluate_model(session, model, evaluation_data):
  count = 1
  x_eval, y_eval = evaluation_data
  total_intersection, total_union = 0., 0.
  for i in range(0, len(x_eval), count):
      end = min(i + count, len(x_eval))
      spliced = x_eval[i:end], y_eval[i:end]
      intersection, union, _ = model.evaluate(session, spliced)
      total_intersection += intersection
      total_union += union
  return intersection, union

def train_and_evaluate(model_gen):
  import galileo.io

  server = 'ec2-54-244-7-55.us-west-2.compute.amazonaws.com:5000'
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
    for i in range(batches):
      print('batch', i + 1, 'of', batches)
      start_load = time.time()
      batch = training_loader.load_batch()
      t = time.time() - start_load
      print('seconds to load batch', t)
      load_time += t
      start_batch = time.time()
      print(model.train_batch(sess, batch))
      t = time.time() - start_batch
      print('seconds to train batch', t)
      batch_time += t
    galileo.io.log_metric('negative_train_time', start_train - time.time())
    galileo.io.log_metadata('average_load_time', load_time / batches)
    galileo.io.log_metadata('average_batch_time', batch_time / batches)
    start_eval = time.time()
    intersection, union = evaluate_model(sess, model, evaluation_data)
    galileo.io.log_metric('negative_eval_time', start_eval - time.time())
    print('final intersection', intersection, 'union', union)
    print('iou', intersection / union)
    galileo.io.log_metadata('intersection', float(intersection))
    galileo.io.log_metadata('union', float(union))
    iou = intersection / union
    galileo.io.log_metadata('iou', float(iou))
    galileo.io.log_metric('loss', 1. - iou)
