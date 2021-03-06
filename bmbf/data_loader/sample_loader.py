from collections import defaultdict
import numpy as np
from PIL import Image

full_image_size = 768

class Segmentation(object):
  def __init__(self, start, run):
    self.start = start
    self.run = run

  def apply_to_image(self, image, value):
    image.reshape((-1, image.shape[-1]))[self.start:self.start+self.run] = value

  def __repr__(self):
    return 'Segmentation(start={}, run={})'.format(self.start, self.run)

class Sample(object):
  def __init__(self, image, segmentation):
    self.image = image
    self.segmentations = [
      Segmentation(start_px, run)
      for start_px, run in zip(segmentation[::2], segmentation[1::2])
    ]

  def load_image(self):
    image_name = 'train/{}'.format(self.image)
    pil_image = Image.open(image_name)
    try:
      pil_image.load()
    except Exception:
      print(image_name)
      raise
    flat_data = np.asarray(pil_image, dtype=np.int32).astype(np.uint8)
    return flat_data.transpose((1, 0, 2)).copy()

  def load_mask(self):
    mask = np.zeros((full_image_size, full_image_size, 1), dtype=np.uint8)
    self.apply_segmentations(mask)
    return mask

  def apply_segmentations(self, image):
    for i, segmentation in enumerate(self.segmentations):
      segmentation.apply_to_image(image, i)

  def __repr__(self):
    return 'Sample(n_segmentations={})'.format(len(self.segmentations))

def load_samples(filename='train_ship_segmentations.csv'):
  with open(filename) as segmentations_fp:
    # skip the header line
    next(segmentations_fp)
    segmentations = defaultdict(list)
    for line in segmentations_fp:
      image, segmentation = line[:-1].split(',')
      segmentations[image].extend(
        list(map(int, segmentation.split(' '))) if segmentation else []
      )
    all_samples = [Sample(image, seg) for image, seg in segmentations.items()]
    return (
      [sample for sample in all_samples if not sample.segmentations],
      [sample for sample in all_samples if sample.segmentations],
    )
