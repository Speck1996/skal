import tensorflow as tf
import numpy as np
import random


@tf.keras.utils.register_keras_serializable()
class DropConcatenate(tf.keras.layers.Concatenate):
    def __init__(self, rate, seed=None, **kwargs):
        self.rate = rate
        self.seed = seed
        super(DropConcatenate, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        "based on dropout rate"

        if training is None:
            training = tf.keras.backend.learning_phase()

        if len(inputs) < 1:
            raise ValueError(
                "A `Concatenate` layer should be called on a list of "
                f"at least 1 input. Received: input_shape={inputs} with len={len(inputs)}"
            )

        if training:
            keep_rate = 1 - self.rate * 1.0 / len(inputs)
            scale = 1.0 / keep_rate

            batch_shape = tf.shape(inputs[0])[0]
            # select number of elements that will be affected by dropout
            rate_rand_uniform = tf.random.uniform(
                [batch_shape, 1], 0, 1, seed=self.seed
            )
            keep_mask = rate_rand_uniform >= self.rate
            keep_mask = tf.cast(keep_mask, dtype=inputs[0].dtype)
            keep_mask = tf.reshape(keep_mask, [-1, 1, 1, 1])
            # continue using tensorflow instead of random library?
            random_value = random.randrange(0, len(inputs))
            masked_input = [
                inputs[i] * keep_mask if i == random_value else inputs[i]
                for i in range(len(inputs))
            ]
            output = super(DropConcatenate, self).call(masked_input) * scale
        else:
            output = super(DropConcatenate, self).call(inputs)
        return output

    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super(DropConcatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable()
class DepthToSpace(tf.keras.layers.Layer):
    """DepthToSpace Layer.

    Rearrenges the depth dimension subdiving it in blocks and rearrenging these blocks
    into the spatial dimension

    """

    def __init__(self, block_size=1, **kwargs):
        self.block_size = block_size
        super(DepthToSpace, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        block_size = self.block_size
        return tf.nn.depth_to_space(input_tensor, block_size)

    def get_config(self):
        config = super(DepthToSpace, self).get_config()
        config.update({"block_size": self.block_size})
        return config


@tf.keras.utils.register_keras_serializable()
class ReflectionPadding2D(tf.keras.layers.Layer):
    """ReflectiionPadding Layer.

    Applies reflectiion padding

    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3],
        )

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(
            input_tensor,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
            "REFLECT",
        )
