from tensorflow import keras
import numpy as np
import tensorflow as tf

from typing import Union


class BaseCustomModel(tf.keras.models.Model):
    """
    This class is an example of how to implement a custom model.
    It is not intended for direct inheritance/usage.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

