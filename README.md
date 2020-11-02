# MaxUp - Data Augmentation Gradient Algorithm in Image Classification


Implementation of [MaxUp](https://arxiv.org/abs/2002.09024) algorithm in image classification


See Colab: https://colab.research.google.com/drive/1KbSZIuiR9hBY2TVaDFCetFaX_tN--4lO?usp=sharing

## Architecture

Sequential = [ResNet50V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2) + 3 stacked layers:
```python
tf.keras.layers.GlobalAveragePooling2D(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation = 'softmax')
```


## Implementation

* Tensorflow 2.0 
* Python 3.7
* Colaboratory

## Dataset

A subset of Imagenet: [Imagenette](https://github.com/fastai/imagenette)

Images are cropped to 256 * 256 (pixels)

## Results

~95.57% on validation dataset using [Categorical Accuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy)


