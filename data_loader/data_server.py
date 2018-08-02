from flask import Flask, request, send_file
import io
import numpy as np

from data_loader import load_samples

app = Flask(__name__)
app.debug = True
samples = load_samples()

class SampleReader(object):
  closed = False

  def __init__(self, batch_size, image_size):
    self.image_size = image_size
    self.samples = np.random.choice(samples, batch_size)
    self.buffers = (self._wrap_buffer(img) for images in (
      (sample.load_image() for sample in self.samples),
      (self._load_mask(sample) for sample in self.samples),
    ) for img in images)
    self.current_buffer = next(self.buffers)

  def _wrap_buffer(self, img):
    buff = io.BytesIO()
    buff.write(memoryview(img))
    buff.seek(0)
    return buff

  def _load_mask(self, sample):
    mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
    sample.apply_segmentations(mask, 1)
    return mask

  def readinto(self, buff):
    data = self.current_buffer.read(len(buff))
    if not data:
      try:
        self.current_buffer = next(self.buffers)
        data = self.current_buffer.read(len(buff))
      except StopIteration:
        pass
    count = len(data)
    buff[:count] = data
    return count

  def readable(self):
    return True

  def close(self):
    pass

  def flush(self):
    pass

@app.route('/train_batch')
def train_batch():
  batch_size = int(request.args.get('batch_size', '1'))
  image_size = int(request.args.get('image_size', '768'))

  return send_file(
    io.BufferedReader(SampleReader(batch_size, image_size)),
    attachment_filename='batch.raw',
    mimetype='application/octet-stream',
  )
