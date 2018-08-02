import numpy as np
from PIL import Image

class Segmentation(object):
  def __init__(self, start, run):
    self.start = start
    self.run = run

  def apply_to_image(self, image, value):
    image.reshape((-1, image.shape[-1]))[self.start:self.start+self.run] = value

  def __repr__(self):
    return 'Segmentation(start={}, run={})'.format(self.start, self.run)

class Sample(object):
  def __init__(self, line):
    self.image, segmentation = line[:-1].split(',')
    segmentation = list(map(int, segmentation.split(' '))) if segmentation else []
    self.segmentations = [Segmentation(start_px, run) for start_px, run in zip(segmentation[::2], segmentation[1::2])]

  def load_image(self):
    pil_image = Image.open('train/{}'.format(self.image))
    pil_image.load()
    image_data = np.asarray(pil_image, dtype=np.int32).transpose((1, 0, 2)).copy()
    return image_data

  def apply_segmentations(self, image, value):
    for segmentation in self.segmentations:
      segmentation.apply_to_image(image, value)

  def __repr__(self):
    return 'Sample(n_segmentations={})'.format(len(self.segmentations))

def load_samples(filename='train_ship_segmentations.csv'):
  with open(filename) as segmentations_fp:
    # skip the header line
    next(segmentations_fp)
    return [Sample(line) for line in segmentations_fp]

if __name__ == '__main__':
  sample = next(iter(load_samples()))
  print(sample)
  print(next(iter(sample.segmentations)))
