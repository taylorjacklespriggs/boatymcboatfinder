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
    multi_fetch=assignments['multi_batch'],
    batch_size=assignments['batch_size'],
    image_size=assignments['image_size'],
    blank_prob=assignments['blank_prob'],
  )
  multi_batch_epochs = assignments['multi_batch_epochs']
  with tf.Session() as sess:
    model = model_gen()
    sess.run(tf.global_variables_initializer())
    start_train = time.time()
    load_time = 0.
    batch_time = 0.
    remaining_time = 60 * 60
    batches = 0

    def log_all_meta():
      if batches:
        galileo.io.log_metadata('average_load_time', load_time / batches)
        galileo.io.log_metadata('average_batch_time', batch_time / batches)
      galileo.io.log_metadata('remaining_time', remaining_time)

    while remaining_time > 0:
      print('remaining time for training', remaining_time)
      start_load = time.time()
      multi_batch = training_loader.load_batch()
      t = time.time() - start_load
      print('seconds to load batch', t)
      load_time += t
      start_batch = time.time()
      try:
        for _ in range(multi_batch_epochs):
          loss = 0
          for batch in multi_batch:
            loss += model.train_batch(sess, batch)
          print('avg loss', loss / len(multi_batch))
      except Exception:
        log_all_meta()
        galileo.io.log_metadata('failure_reason', 'memory')
        raise
      t = time.time() - start_batch
      print('seconds to train batch', t)
      batch_time += t
      remaining_time -= time.time() - start_load
      batches += 1
    galileo.io.log_metadata('train_time', time.time() - start_train)
    log_all_meta()
    start_eval = time.time()
    evaluation_data = load_evaluation_data(server)
    intersection, union = evaluate_model(sess, model, evaluation_data)
    galileo.io.log_metadata('eval_time', time.time() - start_eval)
    print('final intersection', intersection, 'union', union)
    print('iou', intersection / union)
    galileo.io.log_metadata('intersection', float(intersection))
    galileo.io.log_metadata('union', float(union))
    iou = intersection / union
    galileo.io.log_metric('iou', iou)
