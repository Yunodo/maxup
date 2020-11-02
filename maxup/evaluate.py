#Self-Contained script to evaluate model with pre-trained weights

import numpy as np
import tensorflow as tf
import os
import re
from tensorflow import keras

from model import create_model
from load_data import get_paths
from normalise_py import normalise_data

_, validation_path = get_paths()

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  labels = 'inferred',
                                                  label_mode = 'categorical')


validation_dataset = validation_dataset.map(normalise_data)

model = create_model()

model.load_weights('my_model.h5')

model.compile(optimizer = 'adam', loss = None, metrics = keras.metrics.CategoricalAccuracy())

model.evaluate(validation_dataset)
