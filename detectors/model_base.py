import numpy as np
import tensorflow as tf

from constants import assignments, optimizer, x, y, training_mode

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    self.loss = tf.losses.mean_squared_error(y, self.model)
    squared_diff = tf.square(y - self.model)
    blank_pixels = 1. - y
    total_blank_pixels = tf.reduce_sum(blank_pixels)
    self.blank_loss = tf.cond(
      total_blank_pixels > 0.,
      true_fn=lambda: tf.reduce_sum(blank_pixels * squared_diff) / total_blank_pixels,
      false_fn=lambda: 0.,
    )
    total_boat_pixels = tf.reduce_sum(y)
    self.boat_loss = tf.cond(
      total_boat_pixels > 0.,
      true_fn=lambda: tf.reduce_sum(y * squared_diff) / total_boat_pixels,
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
      [self.blank_loss, self.boat_loss],
      feed_dict={x: x_train, y: y_train, training_mode: False}
    )
