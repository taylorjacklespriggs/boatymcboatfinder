import numpy as np
import tensorflow as tf

from constants import assignments, optimizer, x, y, training_mode

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    self.masked_output = y * self.model
    self.intersection = tf.reduce_sum(self.masked_output)
    self.union = tf.reduce_sum(y + self.model - self.masked_output)
    self.iou = self.intersection / self.union
    self.loss = tf.cond(
      self.union > 0.,
      true_fn=lambda: 1. - self.iou,
      false_fn=lambda: self.model,
    )
    self.optimizer = optimizer(
      learning_rate=10**assignments.get('log_learning_rate', -3)
    )
    self.optimizer = self.optimizer.minimize(self.loss)

  def train_batch(self, session, batch):
    x_train, y_train = batch
    return = session.run(
      [self.optimizer, self.loss],
      feed_dict={x: x_train, y: y_train, training_mode: True}
    )[1]

  def forward(self, session, x_data):
    return session.run(
      self.model,
      feed_dict={x: np.expand_dims(x_data, axis=0), training_mode: False}
    )[0]

  def evaluate(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.intersection, self.union],
      feed_dict={x: x_train, y: y_train, training_mode: False}
    )
