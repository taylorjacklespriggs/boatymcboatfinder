import cv2
import math
import numpy as np
import random
import tensorflow as tf
import time

from bmbf.data_loader.data_server import get_training_samples_with_blanks, evaluation_samples
from bmbf.detectors.constants import assignments, x, y, full_size

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

def create_batch(samples):
  imgs = np.stack([sample.load_image() for sample in samples], axis=0)
  X = np.ones((imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3] + 1), dtype=np.float32)
  X[:, :, :, :3] = imgs
  X /= 255
  Y = np.stack([sample.load_mask() for sample in samples], axis=0).astype(np.float32)
  Y[Y > 0] = 1
  return X, Y

evaluation_data = create_batch(evaluation_samples)

def train_and_evaluate(model_gen):
  import orchestrate.io as orc
  orc.log_metadata('start_time', time.time())

  batch_size = assignments.get('batch_size', 1)
  blank_prob = assignments.get('blank_prob', 0)
  with tf.Session() as sess:
    start_train = time.time()
    load_time = 0.
    batch_time = 0.
    remaining_time = assignments.get('training_minutes', 12 * 60) * 60
    batches = 0

    def log_all_meta():
      if batches:
        orc.log_metadata('average_load_time', load_time / batches)
        orc.log_metadata('average_batch_time', batch_time / batches)
      orc.log_metadata('remaining_time', remaining_time)
      orc.log_metadata('batches', batches)

    try:
      model = model_gen()
      sess.run(tf.global_variables_initializer())

      model.evaluate(
        sess,
        (
          np.zeros((batch_size, full_size, full_size, 4)),
          np.zeros((batch_size, full_size, full_size, 1)),
        ),
      )
      model.evaluate(
        sess,
        (
          np.zeros((1, full_size, full_size, 4)),
          np.zeros((1, full_size, full_size, 1)),
        ),
      )
    except Exception:
      log_all_meta()
      orc.log_metadata('failure_reason', 'initialization_memory')
      raise

    while remaining_time > 0:
      loss = 0
      mini_batches = 100
      for _ in range(mini_batches):
        for _ in range(32):
          start_load = time.time()
          samples = get_training_samples_with_blanks(batch_size, blank_prob)
          batch = create_batch(samples)
          t = time.time() - start_load
          load_time += t
          start_batch = time.time()
          try:
            l = model.train_batch(sess, batch)
            loss += l
            print(l, ' '*20, end='\r')
            if remaining_time < time.time() - start_load:
              break
          except Exception:
            log_all_meta()
            orc.log_metadata('failure_reason', 'training_memory')
            raise
          print()
          t = time.time() - start_batch
          batch_time += t
          remaining_time -= time.time() - start_load
          batches += 1
        display_idx = random.randint(0, evaluation_data[0].shape[0] - 1)
        cv2.imshow('image', evaluation_data[0][display_idx])
        cv2.imshow('image_mask', evaluation_data[1][display_idx])
        pred_labels = model.predict(sess, evaluation_data[0][display_idx])
        cv2.imshow('pred_labels', pred_labels)
        output = model.forward(sess, evaluation_data[0][display_idx])
        output /= output.max()
        cv2.imshow('output', output)
        cv2.waitKey(100)
      print()
      print('average loss', loss / mini_batches)
      intersection, union = evaluate_model(sess, model, evaluation_data)
      iou = intersection / union
      print("intersection, union, iou:", intersection, union, iou)
    orc.log_metadata('train_time', time.time() - start_train)
    log_all_meta()
    start_eval = time.time()
    intersection, union = evaluate_model(sess, model, evaluation_data)
    orc.log_metadata('eval_time', time.time() - start_eval)
    print('final intersection', intersection, 'union', union)
    print('iou', intersection / union)
    orc.log_metadata('intersection', float(intersection))
    orc.log_metadata('union', float(union))
    iou = intersection / union
    orc.log_metric('iou', iou)
    orc.log_metadata('finish_time', time.time())
