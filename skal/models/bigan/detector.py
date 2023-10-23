import numpy as np
import tensorflow as tf
from typing import Union

from skal.models.base.detector import AnomalyDetector


class BiganDetector(AnomalyDetector):

    def score_anomalies(self, data: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        data = self.preprocessor(data)
        data_latents = self.model.encoder(data, training=False)
        reconstruction_error = tf.abs(data - self.model.generator(data_latents, training=False))
        discriminator_score = self.model.discriminator([data, data_latents])

        anomaly_scores = self.postprocessor(discriminator_score) * self.postprocessor(reconstruction_error)
        return anomaly_scores