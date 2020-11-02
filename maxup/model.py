#Initialising a new model


def create_model():
  base_model = tf.keras.applications.ResNet50V2(input_shape= IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet') #pre-trained model
  base_model.trainable = True
  model = tf.keras.Sequential([
                               base_model,
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dropout(0.2),
                               tf.keras.layers.Dense(10, activation = 'softmax') # stacking layers on top of pre-trained model
  ])

  return model
