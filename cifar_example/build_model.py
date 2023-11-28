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
from larq import layers

quantized = True

if quantized:
    input_quantizer = 'ste_sign'
    kernel_quantizer = 'ste_sign'
    model_name = 'binarized_net'
else:
    input_quantizer = kernel_quantizer = None
    model_name = 'full_precision_net'
# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

model = keras.models.Sequential([
    # In the first layer we only quantize the weights and not the input
    layers.QuantConv2D(128, 3,
                       kernel_quantizer="ste_sign",
                       kernel_constraint="weight_clip",
                       use_bias=False,
                       input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantConv2D(128, 3, padding="same", **kwargs),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantConv2D(256, 3, padding="same", **kwargs),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantConv2D(256, 3, padding="same", **kwargs),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantConv2D(512, 3, padding="same", **kwargs),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantConv2D(512, 3, padding="same", **kwargs),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),
    keras.layers.Flatten(),

    layers.QuantDense(1024, **kwargs),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantDense(1024, **kwargs),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),

    layers.QuantDense(10, **kwargs),
    keras.layers.BatchNormalization(momentum=0.999, scale=False),
    keras.layers.Activation("softmax")
])

dummy_input = keras.Input((32, 32, 3))
model(dummy_input)

keras.models.save_model(model, f'{model_name}.h5')
