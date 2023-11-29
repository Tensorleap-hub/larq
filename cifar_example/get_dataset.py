import tensorflow as tf
import keras
import numpy as np

from cifar_example.config import CONFIG

np.random.seed(2023)


def get_dataset():
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
    train_labels = np.delete(train_labels, val_indices)
    test_labels = np.asarray(keras.utils.to_categorical(test_labels, num_classes))
    return tf.convert_to_tensor(train_images[:train_size]), \
        tf.convert_to_tensor(train_labels[:train_size]), \
        tf.convert_to_tensor(val_images), \
        tf.convert_to_tensor(val_labels), \
        tf.convert_to_tensor(test_images[:test_size]), \
        tf.convert_to_tensor(test_labels[:test_size])


if __name__ == '__main__':
    _ = get_dataset()
