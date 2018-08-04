import numpy as np
import tensorflow as tf

from constants import assignments, optimizer, x, y, training_mode

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    squared_diff = tf.square(y - self.model)
    blank_pixels = 1. - y
    self.total_blank = tf.reduce_sum(blank_pixels)
    self.error_blank = tf.reduce_sum(blank_pixels * squared_diff)
    self.blank_loss = tf.cond(
      self.total_blank > 0.,
      true_fn=lambda: self.error_blank / self.total_blank,
      false_fn=lambda: 0.,
    )
    self.total_boat = tf.reduce_sum(y)
    self.error_boat = tf.reduce_sum(y * squared_diff)
    self.boat_loss = tf.cond(
      self.total_boat > 0.,
      true_fn=lambda: self.error_boat / self.total_boat,
      false_fn=lambda: 0.,
    )
    self.loss = self.boat_loss + self.blank_loss
    self.optimizer = optimizer(
      learning_rate=10**assignments.get('log_learning_rate', -3)
    )
    self.optimizer = self.optimizer.minimize(self.loss)

  def train_batch(self, session, batch):
    x_train, y_train = batch
    _, loss = session.run(
      [self.optimizer, self.loss],
      feed_dict={x: x_train, y: y_train, training_mode: True}
    )
    return loss

  def forward(self, session, x_data):
    return session.run(
      self.model,
      feed_dict={x: np.expand_dims(x_data, axis=0), training_mode: False}
    )[0]

  def evaluate(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.error_blank, self.total_blank, self.error_boat, self.total_boat],
      feed_dict={x: x_train, y: y_train, training_mode: False}
    )
