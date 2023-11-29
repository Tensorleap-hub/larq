from typing import List

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse

from cifar_example.config import CONFIG
from larq import (
    activations,
    callbacks,
    constraints,
    context,
    layers,
    math,
    metrics,
    models,
    optimizers,
    quantizers,
    utils,
)  # needed imports
from larq.layers import QuantDense, QuantConv2D
import tensorflow as tf
import keras

np.random.seed(2023)


# Preprocess Function
def get_responses():
    num_classes = CONFIG['n_classes']
    train_size = CONFIG['train_size']
    val_size = CONFIG['val_size']
    test_size = CONFIG['test_size']

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    train_images = np.asarray(train_images.reshape((50000, 32, 32, 3)).astype("float32"))
    val_indices = np.random.choice(np.arange(len(train_images)), val_size, replace=False)

    val_images = train_images[val_indices, ...]
    train_images = np.delete(train_images, val_indices, axis=0)
    test_images = np.asarray(test_images.reshape((10000, 32, 32, 3)).astype("float32"))

    # Normalize pixel values to be between -1 and 1
    train_images, val_images, test_images = train_images / 127.5 - 1, val_images / 127.5 - 1, test_images / 127.5 - 1

    train_labels = np.asarray(keras.utils.to_categorical(train_labels, num_classes))
    val_labels = train_labels[val_indices]
    train_labels = np.delete(train_labels, val_indices, axis=0)
    test_labels = np.asarray(keras.utils.to_categorical(test_labels, num_classes))

    train = PreprocessResponse(length=val_images.shape[0], data={'images': train_images[:train_size], 'labels': train_labels[:train_size]})
    val = PreprocessResponse(length=train_images.shape[0], data={'images': val_images, 'labels': val_labels})
    test = PreprocessResponse(length=test_images.shape[0], data={'images': test_images[:test_size], 'labels': test_labels[:test_size]})
    return [train, val, test]


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.asarray(preprocess.data['images'][idx].astype('float32'))


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.asarray(preprocess.data['labels'][idx].astype('float32'))


def placeholder_loss(y_true, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
    return tf.reduce_mean(y_true, axis=-1) * 0


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=get_responses)
leap_binder.set_input(function=input_encoder, name='input')
leap_binder.set_ground_truth(function=gt_encoder, name='gt')
leap_binder.set_custom_layer(QuantDense, 'QuantDense')
leap_binder.set_custom_layer(QuantConv2D, 'QuantConv2D')
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')

if __name__ == '__main__':
    leap_binder.check()
