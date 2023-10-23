from tensorflow import keras
import numpy as np
import tensorflow as tf
from typing import Union


class AnomalyDetector:
    def __init__(self, preprocessor, model, postprocessor) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def score_anomalies(self, data: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        raise NotImplementedError
