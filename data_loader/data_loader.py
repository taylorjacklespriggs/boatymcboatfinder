class Segmentation(object):
  def __init__(self, start, run):
    self.start = start
    self.run = run

  def __repr__(self):
    return 'Segmentation(start={}, run={})'.format(self.start, self.run)

class Sample(object):
  def __init__(self, line):
    self.image, segmentation = line[:-1].split(',')
    segmentation = list(map(int, segmentation.split(' ')))
    self.segmentations = [Segmentation(start_px, run) for start_px, run in zip(segmentation[::2], segmentation[1::2])]

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
