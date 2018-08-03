import tensorflow as tf

from constants import assignments, optimizer, x, y, training_mode

class ModelBase(object):
  def __init__(self):
    self.model = self.create_model()
    self.loss = tf.losses.mean_squared_error(y, self.model)
    self.optimizer = optimizer(learning_rate=10**assignments.get('log_learning_rate', -3))
    self.optimizer = self.optimizer.minimize(self.loss)

  def train_batch(self, session, batch):
    x_train, y_train = batch
    _, loss = session.run(
      [self.optimizer, self.loss],
      feed_dict={x: x_train, y: y_train, training_mode: True}
    )
    return loss

  def forward(self, session, x_data):
    return session.run(self.model, feed_dict={x: x_data, training_mode: False})

  def evaluate(self, session, batch):
    x_train, y_train = batch
    return session.run(
      self.loss,
      feed_dict={x: x_train, y: y_train, training_mode: False}
    )
