import numpy as np

from bmbf.data_loader.sample_loader import load_samples

cv_parts = 400
blank_samples, boat_samples = load_samples()
blank_split = len(blank_samples) // cv_parts
boat_split = len(boat_samples) // cv_parts
evaluation_samples, blank_training_samples, boat_training_samples = (
  blank_samples[:blank_split] + boat_samples[:boat_split],
  blank_samples[blank_split:],
  boat_samples[boat_split:],
)

def get_training_samples_with_blanks(batch_size, blank_prob):
  all_training = blank_training_samples + boat_training_samples
  p = np.zeros((len(all_training),), dtype=np.float32)
  n_blank = len(blank_training_samples)
  p[:n_blank] = blank_prob / n_blank
  n_boat = len(boat_training_samples)
  p[n_blank:] = (1. - blank_prob) / n_boat
  return np.random.choice(all_training, batch_size, p=p)
