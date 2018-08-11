import numpy as np
from urllib.request import urlopen

def read_batch(fp):
  count, rows, cols = np.frombuffer(fp.read(12), dtype=np.uint32).astype(int)
  batch_pixels = count * rows * cols
  x_pixels = np.frombuffer(fp.read(batch_pixels * 3), dtype=np.uint8)
  x_pixels = x_pixels.reshape((count, rows, cols, 3))
  x = np.zeros((count, rows, cols, 4), dtype=np.float32)
  x[:, :, :, :3] = x_pixels
  x /= 255
  y = np.frombuffer(fp.read(batch_pixels), dtype=np.uint8)
  y = y.reshape((count, rows, cols, 1)).astype(np.float32)
  return x, y

class TrainingDataLoader(object):
  def __init__(self, server, multi_fetch, batch_size, image_size, blank_prob):
    self.server = server
    self.multi_fetch = multi_fetch
    self.batch_size = batch_size
    self.image_size = image_size
    self.blank_prob = blank_prob

  def get_train_url(self):
    return 'http://{}/train_batch?batch_size={}&image_size={}&blank_prob={}'.format(
      self.server,
      self.multi_fetch * self.batch_size,
      self.image_size,
      self.blank_prob,
    )

  def load_batch(self):
    with urlopen(self.get_train_url()) as train_fp:
      x, y = read_batch(train_fp)
      count, rows, cols, chan = x.shape
      x = x.reshape((self.multi_fetch, count // self.multi_fetch, rows, cols, chan))
      y = y.reshape((self.multi_fetch, count // self.multi_fetch, rows, cols, 1))
      return list(zip(x, y))

def load_evaluation_data(server):
  url = 'http://{}/evaluation_batch'.format(server)
  with urlopen(url) as eval_fp:
    return read_batch(eval_fp)
