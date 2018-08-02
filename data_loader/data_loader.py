class Sample(object):
  def __init__(self, line):
    self.image, segmentation = line[:-1].split(',')
    segmentation = list(map(int, data.split(' ')))
    self.segmentations = list(zip(segmentation[::2], segmentation[1::2]))

def load_segmentations(filename='train_ship_segmentations.csv'):
  with open(filename) as segmentations_fp:
    training_samples = [Sample(line) for line in segmentations_fp]
