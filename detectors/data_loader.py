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

  def load_batch(self):
    with urlopen(self.get_train_url()) as train_fp:
      x = np.frombuffer(train_fp.read(self.batch_pixels * 3), dtype=np.uint8)
      x = x.reshape((self.batch_size, self.image_size, self.image_size, 3)).astype(np.float32)
      x /= 255
      y = np.frombuffer(train_fp.read(self.batch_pixels), dtype=np.uint8)
      y = y.reshape((self.batch_size, self.image_size, self.image_size, 1)).astype(np.float32)
      return x, y

def load_evaluation_data(server):
  url = 'http://{}/evaluation_batch'.format(self.server)
  full_size = 768

  with urlopen(url) as eval_fp:
    count = np.frombuffer(eval_fp.read(4), dtype=np.uint32)[0]
    x = np.frombuffer(eval_fp.read(count * 3), dtype=np.uint8)
    x = x.reshape((count, full_size, full_size, 3)).astype(np.float32)
    x /= 255
    y = np.frombuffer(train_fp.read(count), dtype=np.uint8)
    y = y.reshape((count, full_size, full_size, 1)).astype(np.float32)
    return x, y
