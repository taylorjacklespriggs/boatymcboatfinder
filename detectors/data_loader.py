import numpy as np
from urllib.request import urlopen

def read_batch(fp):
  count, rows, cols = np.frombuffer(fp.read(12), dtype=np.uint32).astype(int)
  batch_pixels = count * rows * cols
  x = np.frombuffer(fp.read(batch_pixels * 3), dtype=np.uint8)
  x = x.reshape((count, rows, cols, 3)).astype(np.float32)
  x /= 255
  y = np.frombuffer(fp.read(batch_pixels), dtype=np.uint8)
  y = y.reshape((count, rows, cols, 1)).astype(np.float32)
  return x, y

class TrainingDataLoader(object):
  def __init__(self, server, batch_size, image_size):
    self.server = server
    self.batch_size = batch_size
    self.image_size = image_size

  def get_train_url(self):
    return 'http://{}/train_batch?batch_size={}&image_size={}'.format(
      self.server,
      self.batch_size,
      self.image_size,
    )

  def load_batch(self):
    with urlopen(self.get_train_url()) as train_fp:
      return read_batch(train_fp)

def load_evaluation_data(server):
  url = 'http://{}/evaluation_batch'.format(server)
  with urlopen(url) as eval_fp:
    return read_batch(eval_fp)
