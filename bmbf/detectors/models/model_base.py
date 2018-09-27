from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from bmbf.detectors.constants import assignments, batch_size, optimizer, x, y, training_mode, learning_rate

def safe_log(value):
  return tf.cond(
    value > 0.,
    true_fn=lambda: tf.log(value),
    false_fn=lambda: 0.,
  )

def threshhold(tensor, thresh=1):
  return tf.cast(tensor > thresh, tf.float32)

def iou_loss(labels, pred):
  correct_pred = tf.reduce_sum(labels * pred)
  total_pred = tf.reduce_sum(pred)
  return tf.log(total_pred + tf.reduce_sum(labels) - correct_pred) - tf.log(correct_pred)

class ModelBase(object):
  def __init__(self):
    self.input = Input(shape=(768, 768, 4))
    self.model = self.create_model()
    self.model.compile(
      optimizer=optimizer,
      loss=iou_loss,
      metrics=['accuracy'],
    )

  def train(self, batch_gen, evaluation_data):
    return self.model.fit_generator(
      generator=batch_gen,
      steps_per_epoch=10,
      validation_data=evaluation_data,
    )

  def forward(self, x_data):
    return self.model.predict(
      x_data[np.newaxis],
      self.model,
    )[0]
