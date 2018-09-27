from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from bmbf.detectors.constants import optimizer

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

def iou(labels, pred):
  binary_pred = pred > 0.5
  intersection = tf.reduce_sum(labels * binary_pred)
  return tf.reduce_sum(labels) + tf.reduce_sum(binary_pred) - intersection

class ModelBase(object):
  def __init__(self):
    self.input = Input(shape=(768, 768, 4))
    self.model = Model(inputs=self.input, outputs=self.create_model())
    self.model.compile(
      optimizer=optimizer,
      loss=iou_loss,
      metrics=[iou],
    )

  def train(self, batch_gen, steps, evaluation_data):
    return self.model.fit_generator(
      generator=batch_gen,
      steps_per_epoch=steps,
      validation_data=evaluation_data,
    )

  def forward(self, x_data):
    return self.model.predict(
      x_data[np.newaxis],
      self.model,
    )[0]
