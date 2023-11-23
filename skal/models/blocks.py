import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from typing import Union


def ConvBlock(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "valid",
    dilation_rate=(1, 1),
    activation: Union[str, tf.keras.layers.Layer] = None,
    use_bias: bool = False,
    kernel_initializer="glorot_uniform",
    apply_normalization: bool = True,
    drop_rate: float = 0.1,
    seed: int = None,
):
    def apply(x):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
        )(x)

        if apply_normalization:
            x = keras.layers.GroupNormalization(groups=16)(x)

        if isinstance(activation, str):
            x = keras.layers.Activation(activation)(x)
        elif issubclass(
            type(activation), (keras.layers.LeakyReLU, keras.layers.ReLU)
        ):
            x = activation(x)
        elif activation is not None:
            raise ValueError(f"Unknown activation {activation}")

        if drop_rate > 0.0 and drop_rate < 1.0:
            x = keras.layers.SpatialDropout2D(drop_rate, seed=seed)(x)

        return x

    return apply


def ResBlock(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    dilation_rate=(1, 1),
    kernel_initializer="glorot_uniform",
    drop_rate: float = 0.0,
    seed: int = None,
):
    def apply(x):
        x_channels = x.shape[-1]
        shortcut = x

        # Convolutional block
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            dilation_rate=dilation_rate,
            use_bias=False,
            kernel_initializer=kernel_initializer,
        )(x)
        x = tfa.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SpatialDropout2D(drop_rate, seed=seed)(x)  # Add dropout
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            dilation_rate=dilation_rate,
        )(x)
        x = tfa.layers.GroupNormalization(groups=8)(x)

        # adjusting Shortcut connection when the conv block has different filters than the input
        if not x_channels == filters:
            shortcut = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=strides,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tfa.layers.GroupNormalization(groups=8)(shortcut)

        residual = tf.keras.layers.Add()([x, shortcut])
        output = tf.keras.layers.Activation("relu")(residual)
        return output

    return apply
