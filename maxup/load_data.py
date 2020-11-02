#Downloading data

def get_paths():
  _URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
  path_to_zip = tf.keras.utils.get_file('imagenette2-320.tgz', origin=_URL, extract=True)
  PATH = os.path.join(os.path.dirname(path_to_zip), 'imagenette2-320')

  train_dir = os.path.join(PATH, 'train')
  validation_dir = os.path.join(PATH, 'val')

  return train_dir, validation_dir
