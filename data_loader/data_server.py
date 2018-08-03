from flask import Flask, request, send_file
import io
import numpy as np

from data_loader import load_samples

app = Flask(__name__)
app.debug = True
samples = load_samples()
split = len(samples) // 100
print(split)
evaluation_samples, training_samples = samples[:split], samples[split:]

full_image_size = 768

def load_mask(sample):
  mask = np.zeros((full_image_size, full_image_size, 1), dtype=np.uint8)
  sample.apply_segmentations(mask, 1)
  return mask


class Subsample(object):
  def __init__(self, image_size, sample):
    self.sample = sample
    self.image_size = image_size
    center_index = np.random.choice(np.concatenate(tuple(
      np.arange(segment.start, segment.start+segment.run)
      for segment in sample.segmentations
    )))
    center_index = center_index // full_image_size, center_index % full_image_size
    (self.i_min, self.i_max), (self.j_min, self.j_max) = map(self._get_boundaries, center_index)

  def _get_boundaries(self, idx):
    idx -= self.image_size // 2
    start = max(0, min(idx, full_image_size - self.image_size))
    end = start + self.image_size
    return start, end

  def _splice(self, img):
    return img[self.i_min:self.i_max, self.j_min:self.j_max].copy()

  def load_image(self):
    return self._splice(self.sample.load_image())

  def load_mask(self):
    return self._splice(load_mask(self.sample))


class ImageReader(object):
  closed = False

  def __init__(self, shape, image_gen):
    self.buffers = (self._wrap_buffer(img) for img in image_gen)
    self.current_buffer = self._wrap_buffer(np.array(shape, dtype=np.uint32))

  def _wrap_buffer(self, img):
    buff = io.BytesIO()
    buff.write(memoryview(img))
    buff.seek(0)
    return buff

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

  subsamples = [
    Subsample(image_size, sample)
    for sample in np.random.choice(training_samples, batch_size)
  ]
  images = (img for images in (
    (sample.load_image() for sample in subsamples),
    (sample.load_mask() for sample in subsamples),
  ) for img in images)

  return send_file(
    io.BufferedReader(ImageReader((batch_size, image_size, image_size), images)),
    attachment_filename='batch.raw',
    mimetype='application/octet-stream',
  )

@app.route('/evaluation_batch')
def evaluation_batch():
  images = (img for images in (
    (sample.load_image() for sample in evaluation_samples),
    (load_mask(sample) for sample in evaluation_samples),
  ) for img in images)
  return send_file(
    io.BufferedReader(ImageReader((len(evaluation_samples), 768, 768), images)),
    attachment_filename='batch.raw',
    mimetype='application/octet-stream',
  )

if __name__ == '__main__':
  app.run(host='0.0.0.0')
