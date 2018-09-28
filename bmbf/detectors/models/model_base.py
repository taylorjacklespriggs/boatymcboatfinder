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

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    self.predicted_labels = threshhold(self.model)
    self.soft_predicted_labels = tf.tanh(self.model)
    self.masked_prediction = y * self.model
    self.intersection = tf.reduce_sum(threshhold(self.masked_prediction))
    self.union = tf.reduce_sum(y) + tf.reduce_sum(self.predicted_labels) - self.intersection
    self.iou = self.intersection / self.union

    self.sum_masked_prediction = tf.reduce_sum(self.masked_prediction)
    self.sum_prediction = tf.reduce_sum(self.model) + tf.reduce_sum(y) - self.sum_masked_prediction
    self.log_sum_prediction = safe_log(self.sum_prediction)
    self.log_sum_masked_prediction = safe_log(self.sum_masked_prediction)
    self.loss = self.log_sum_prediction - self.log_sum_masked_prediction

    self.optimizer = optimizer(learning_rate=learning_rate)
    self.optimizer = self.optimizer.minimize(self.loss)

  def train_batch(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.optimizer, self.loss],
      feed_dict={
        batch_size: x_train.shape[0],
        x: x_train,
        y: y_train,
        training_mode: True,
        learning_rate: 10**assignments.get('log_learning_rate', -3),
      }
    )[1]

  def forward(self, session, x_data):
    return session.run(
      self.model,
      feed_dict={batch_size: 1, x: np.expand_dims(x_data, axis=0), training_mode: False}
    )[0]

  def predict(self, session, x_data):
    return session.run(
      self.soft_predicted_labels,
      feed_dict={batch_size: 1, x: np.expand_dims(x_data, axis=0), training_mode: False}
    )[0]

  def evaluate(self, session, batch):
    x_train, y_train = batch
    return session.run(
      [self.intersection, self.union, self.loss],
      feed_dict={batch_size: x_train.shape[0], x: x_train, y: y_train, training_mode: False}
    )
