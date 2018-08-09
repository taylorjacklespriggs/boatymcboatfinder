import numpy as np
import tensorflow as tf

from constants import assignments, optimizer, x, y, training_mode, learning_rate

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    self.masked_output = y * self.model
    self.intersection = tf.reduce_sum(self.masked_output)
    log_intersection = tf.cond(
      self.intersection > 0.,
      true_fn=tf.log(self.intersection),
      false_fn=tf.constant(0., dtype=tf.float64),
    )
    self.union = tf.reduce_sum(y + self.model - self.masked_output)
    log_union = tf.cond(
      self.union > 0.,
      true_fn=tf.log(self.union),
      false_fn=tf.constant(0., dtype=tf.float64),
    )
    self.iou = self.intersection / self.union
    self.loss = log_union - log_intersection
    self.optimizer = optimizer(learning_rate=learning_rate)
    self.optimizer = self.optimizer.minimize(self.loss)

  def train_batch(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.optimizer, self.loss],
      feed_dict={
        x: x_train,
        y: y_train,
        training_mode: True,
        learning_rate=10**assignments.get('log_learning_rate', -3),
      }
    )[1]

  def forward(self, session, x_data):
    return session.run(
      self.model,
      feed_dict={x: np.expand_dims(x_data, axis=0), training_mode: False}
    )[0]

  def evaluate(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.intersection, self.union, self.loss],
      feed_dict={x: x_train, y: y_train, training_mode: False}
    )
