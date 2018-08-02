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
    self.samples = iter(np.random.choice(samples, batch_size))
    self.image_size = image_size
    self._load_next_image()

  def _load_next_image(self):
    self.stream = io.BytesIO()
    try:
      sample = next(self.samples)
    except StopIteration:
      return
    image = sample.load_image()
    self.stream.write(memoryview(image))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    sample.apply_segmentations(mask, 1)
    self.stream.write(memoryview(mask))
    self.stream.seek(0)

  def read(size=-1):
    data = self.stream.read(size)
    if size and not data:
      self._load_next_image()
    return data

  def readable(self):
    return True


@app.route('/train_batch')
def train_batch():
  batch_size = int(request.args.get('batch_size', '1'))
  image_size = int(request.args.get('image_size', '768'))

  return send_file(
    io.BufferedReader(SampleReader(batch_size, image_size)),
    attachment_filename='batch.raw',
    mimetype='application/octet-stream',
  )
