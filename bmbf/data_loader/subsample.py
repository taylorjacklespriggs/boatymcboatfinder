import numpy as np

class Subsample(object):
  def __init__(self, image_size, sample):
    self.sample = sample
    self.image_size = image_size
    if sample.segmentations:
      center_index = np.random.choice(np.concatenate(tuple(
        np.arange(segment.start, segment.start+segment.run)
        for segment in sample.segmentations
      )))
    else:
      center_index = random.randint(0, full_image_size**2-1)
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
    return self._splice(load_mask(self.sample)())
