import collections
import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent

from tensorflow_federated import python as tff
from random import choices

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 3

tf.compat.v1.enable_v2_behavior()

# Loading simulation data
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):
  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)


def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]
