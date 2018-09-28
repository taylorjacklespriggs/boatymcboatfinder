from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import numpy as np
import tensorflow as tf
import os

from bmbf.detectors.constants import optimizer

gpus = int(os.environ.get('GPUS', '0'))

def safe_log(value):
  return tf.cond(
    value > 0.,
    true_fn=lambda: tf.log(value),
    false_fn=lambda: 0.,
  )

def iou_loss(labels, pred):
  correct_pred = tf.reduce_sum(labels * pred)
  total_pred = tf.reduce_sum(pred)
  return safe_log(total_pred + tf.reduce_sum(labels) - correct_pred) - safe_log(correct_pred)

def intersection(labels, pred):
  return tf.reduce_sum(tf.cast(labels * pred > 0.5, tf.float32))

def union(labels, pred):
  return tf.reduce_sum(labels) \
    + tf.reduce_sum(tf.cast(pred > 0.5, tf.float32)) \
    - intersection(labels, pred)

def iou(labels, pred):
  return intersection(labels, pred) / union(labels, pred)

class ModelBase(object):
  def __init__(self):
    self.input = Input(shape=(768, 768, 4))
    self.model = Model(inputs=self.input, outputs=self.create_model())
    if gpus > 1:
      self.model = multi_gpu_model(self.model, gpus=gpus)
    self.model.compile(
      optimizer=optimizer,
      loss=iou_loss,
      metrics=[intersection, union, iou],
    )

  def train(self, batch_gen, steps):
    return self.model.fit_generator(
      generator=batch_gen,
      steps_per_epoch=steps,
    )

  def forward(self, x_data):
    return self.model.predict(
      x_data[np.newaxis],
      self.model,
    )[0]

  def evaluate(self, evaluation_data, batch_size=1):
    x_data, y_data = evaluation_data
    return self.model.evaluate(
      x=x_data,
      y=y_data,
      batch_size=batch_size,
    )
