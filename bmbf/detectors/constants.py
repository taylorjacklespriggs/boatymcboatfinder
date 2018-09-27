from keras import backend as K
from keras.layers import Concatenate, Dot, Input
from keras.optimizers import Adam
import orchestrate.io as orch

full_size = 768

optimizer = Adam(lr=orch.assignment('log_learning_rate', -3))
