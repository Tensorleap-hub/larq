import os
from keras.callbacks import ModelCheckpoint

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
import keras
from get_dataset import get_dataset
from larq.layers import QuantConv2D, QuantDense

if __name__ == '__main__':
    model_name = 'full_precision_net.h5'
    root_dir = os.path.dirname(os.getcwd())
    model = keras.models.load_model(f'{root_dir}/model/{model_name}', custom_objects={'quantconv': QuantConv2D,
                                                                                      'quantdense': QuantDense})
    model.compile(
        keras.optimizers.Adam(learning_rate=0.01, weight_decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_images, train_labels, val_images, val_labels, test_images, test_labels = get_dataset()

    checkpoint = ModelCheckpoint(
        f"{root_dir}/model/trained_{model_name}",  # Filepath to save the best model
        monitor='val_accuracy',  # Metric to monitor for improvement
        save_best_only=True,  # Save only the best model
        mode='max',  # Mode for monitoring ('max' means save the model when the monitored metric is maximized)
        verbose=1  # Display more information about the saving process
    )

    model.fit(
        train_images,
        train_labels,
        batch_size=64,
        epochs=100,
        validation_data=(val_images, val_labels),
        shuffle=True,
        callbacks=[checkpoint]
    )

    best_model = keras.models.load_model(f"{root_dir}/model/trained_{model_name}",
                                         custom_objects={'quantconv': QuantConv2D,
                                                         'quantdense': QuantDense})
    test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
