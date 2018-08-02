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
    self.samples = (self._load_next_image(sample) for sample in np.random.choice(samples, batch_size))
    self.current_buffer = next(self.samples)

  def _load_next_image(self, sample):
    stream = io.BytesIO()
    image = sample.load_image()
    stream.write(memoryview(image))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    sample.apply_segmentations(mask, 1)
    stream.write(memoryview(mask))
    stream.seek(0)
    return stream

  def readinto(self, buff):
    data = self.current_buffer.read(len(buff))
    if not data:
      try:
        self.current_buffer = next(self.samples)
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
