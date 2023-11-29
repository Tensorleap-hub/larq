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
    model = keras.models.load_model(f'../model/{model_name}', custom_objects={'quantconv': QuantConv2D,
                                                                              'quantdense': QuantDense})
    model.compile(
        keras.optimizers.Adam(learning_rate=0.01, weight_decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_images, train_labels, test_images, test_labels = get_dataset()

    checkpoint = ModelCheckpoint(
        f"../model/trained_{model_name}",  # Filepath to save the best model
        monitor='val_accuracy',  # Metric to monitor for improvement
        save_best_only=True,  # Save only the best model
        mode='max',  # Mode for monitoring ('max' means save the model when the monitored metric is maximized)
        verbose=1  # Display more information about the saving process
    )

    trained_model = model.fit(
        train_images,
        train_labels,
        batch_size=50,
        epochs=1,
        validation_data=(test_images, test_labels),
        shuffle=True,
        callbacks=[checkpoint]
    )
