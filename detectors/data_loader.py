import numpy as np
from urllib.request import urlopen

class TrainingDataLoader(object):
  def __init__(self, server, batch_size, image_size):
    self.server = server
    self.batch_size = batch_size
    self.image_size = image_size
    self.batch_pixels = self.batch_size * self.image_size**2

  def get_train_url(self):
    return 'http://{}/train_batch?batch_size={}&image_size={}'.format(
      self.server,
      self.batch_size,
      self.image_size,
    )

  def load_train_batch(self):
    with urlopen(self.get_train_url()) as train_fp:
      x = np.frombuffer(train_fp.read(self.batch_pixels * 3), dtype=np.int8)
      x = x.reshape((self.batch_size, self.image_size, self.image_size, 3)).copy()
      y = np.frombuffer(train_fp.read(self.batch_pixels), dtype=np.int8)
      y = y.reshape((self.batch_size, self.image_size, self.image_size, 1)).copy()
      return x, y
