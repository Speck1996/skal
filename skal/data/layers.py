import tensorflow as tf
from tensorflow import keras
from functools import partial
from typing import Tuple

from skal.ops import images_tfops


class Tiling(keras.layers.Layer):
    def __init__(self, patch_size: Tuple):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, inputs):
        if len(tf.shape(inputs)) == 4:
            tiling_fn = partial(images_tfops.tile_image, patch_size=self.patch_size)
            return tf.vectorized_map(tiling_fn, inputs)
        else:
            return images_tfops.tile_image(inputs, patch_size=self.patch_size)
        

class Untiling(keras.layers.Layer):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
    
    def call(self, inputs):
        if len(tf.shape(inputs)) == 4:
            untiling_fn = partial(images_tfops.untile_image, image_shape=self.image_size)
            return tf.vectorized_map(untiling_fn, inputs)
        else:
            return images_tfops.untile_image(inputs, image_shape=self.image_size)


class RandomPatches(keras.layers.Layer):
    def __init__(self, patch_size: int, seed:int = None):
        super().__init__()
        self.patch_size = patch_size
        self.seed = seed

    def call(self, inputs):
        # TODO random patch extraction
        size = [*self.patch_size, tf.shape(inputs)[-1]]

        if len(tf.shape(inputs)) == 3:
            inputs = tf.expand_dims(inputs, axis=0)

            patches = tf.vectorized_map(lambda x: tf.image.random_crop(x, size=size, seed=self.seed), inputs)

            return patches
        else:
            return tf.image.random_crop(inputs, size=size, seed=self.seed)


class ResizePadCrop(keras.layers.Layer):
    def __init__(self, target_height: int, target_width: int):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        if len(tf.shape(inputs)) == 4:
            resize_fn = partial(tf.image.resize_with_crop_or_pad, 
                                target_height=self.target_height, target_width=self.target_width)
            return tf.vectorized_map(resize_fn, inputs)
        else:
            return images_tfops.tile_image(inputs, patch_size=self.patch_size)
