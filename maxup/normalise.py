#Normalise data

def normalise_data(image, cl):
  """

  Inputs:
    image - Tensor of image matrix values
    cl    - Tensor of class values

  Outputs:
    normalized - Tensor of normalized image matrix values
    cl         - Tensor of class values

  """

  normalized = tf.cast(image, tf.float32) / 255.0 # convert each 0-255 value to floats in [0, 1] range
  return normalized, cl
