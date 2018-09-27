import math
import numpy as np
import random
import time

from bmbf.data_loader.data import get_training_samples_with_blanks, evaluation_samples
from bmbf.detectors.constants import assignments

input_transform = np.array([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
], dtype=np.float32) / 255

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
  X = np.einsum(
    'bhwc,co->bhwo',
    np.stack([sample.load_image() for sample in samples], axis=0),
    input_transform,
  )
  Y = np.stack([sample.load_mask() for sample in samples], axis=0).astype(np.float32)
  Y[Y > 0] = 1
  return X, Y

def batch_gen(batch_size, blank_prob):
  while True:
    samples = get_training_samples_with_blanks(batch_size, blank_prob)
    yield create_batch(samples)


def train_and_evaluate(model_gen):
  import orchestrate.io as orc
  orc.log_metadata('start_time', time.time())

  batch_size = assignments.get('batch_size', 4)
  blank_prob = assignments.get('blank_prob', 0.2)
  train_steps = assignments.get('train_steps', 1)

  model = model_gen()

  model.train(batch_gen(batch_size, blank_prob), train_steps)

  evaluation_data = create_batch(evaluation_samples)
  print(model.evaluate(evaluation_data))
