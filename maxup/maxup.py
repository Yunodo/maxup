def MaxUp(model, x_batch_train, y_batch_train, loss_fn, augmentations, times):
  """MaxUp (https://arxiv.org/abs/2002.09024) implementation of choosing augmentations with highest loss

    Args:
        model
        x_batch_train
        y_batch_train
        loss_fn
        augmentations: a list of augmentation functions
        times: - the number of times augmentations apply


    Returns:
        Training batch with augmented images with the highest loss
    """

  x_batch = x_batch_train.numpy()
  batch_size = tf.shape(x_batch_train)[0]

  for i in range(batch_size):

    x = tf.expand_dims(x_batch_train[i], axis = 0)
    y = tf.expand_dims(y_batch_train[i], axis = 0)
    logits = model(x, training = False)
    loss_value = loss_fn(y, logits)

    for j in range(0, times):

      for f in augmentations:

        new_x = tf.expand_dims(tf.clip_by_value(f(x_batch_train[i]),0,1), axis = 0)
        logits = model(new_x, training = False)
        new_loss_value = loss_fn(y, logits)

        if new_loss_value > loss_value:

          loss_value = new_loss_value
          x_batch[i] = new_x

  return tf.convert_to_tensor(x_batch)
