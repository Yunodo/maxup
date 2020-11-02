import numpy as np
import tensorflow as tf
import os
import re
from tensorflow import keras


from normalise import normalise_data
from load_data import get_paths
from model import create_model

#Connecting to GPU in Colaboratory

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

#Dataset Parameters

BATCH_SIZE = 16
IMG_SIZE = (256, 256)

#Load Dataset
training_path, validation_path = get_paths()

#Preparing datasets
training_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             labels = 'inferred',
                                             label_mode = 'categorical')

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  labels = 'inferred',
                                                  label_mode = 'categorical')

training_dataset = training_dataset.map(normalise_data)
validation_dataset = validation_dataset.map(normalise_data)


#Creating

model = create_model()

#Create augmentations
from augmentations import flip, color, rotate
augmentations = [flip, color, rotate]

#Compile model
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

metrics = ['Categorical_Accuracy', 'Categorical_Crossentropy']

model.compile(optimizer, loss_fn, metrics)

train_acc_metric = keras.metrics.CategoricalAccuracy()

#Train model
epochs = 3
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))


    for step, (x_batch_train, y_batch_train) in enumerate(training_dataset):
      augmented_x_batch_train = MaxUp(model, x_batch_train, y_batch_train, loss_fn, augmentations, times = 2)


      with tf.GradientTape() as tape:


          logits = model(augmented_x_batch_train, training=True)


          loss_value = loss_fn(y_batch_train, logits)

      grads = tape.gradient(loss_value, model.trainable_weights)


      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      train_acc_metric.update_state(y_batch_train, logits)


      if step % 10 == 0:
          model.save_weights("ckpt")
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step, float(loss_value))
          )
          print("Seen so far: %s samples" % ((step + 1) * 16))
          print("accuracy {:1.2f}".format(train_acc_metric.result().numpy()))

    train_acc_metric.reset_states()

#Evaluate model

model.evaluate(validation_dataset)
