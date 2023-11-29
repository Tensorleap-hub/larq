import keras


def get_dataset():
    num_classes = 10
    train_size = 5000
    test_size = 1000

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")[:train_size, ...]
    test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")[:test_size, ...]

    # Normalize pixel values to be between -1 and 1
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

    train_labels = keras.utils.to_categorical(train_labels, num_classes)[:train_size, ...]
    test_labels = keras.utils.to_categorical(test_labels, num_classes)[:test_size, ...]
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    _ = get_dataset()
