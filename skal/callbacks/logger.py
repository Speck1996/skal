import math
import tensorflow as tf
from tensorflow import keras

from skal.experiment import loggers
from skal.experiment.config import Config
from skal.visualization import plotters


class ExperimentLogger(keras.callbacks.Callback):
    """Callback needed to track hyperparameters"""
    def __init__(
        self,
        log_dir: str,
        exp_config: Config = None,
    ):
        super().__init__()
        self.logger = loggers.TensorboardLogger(log_dir)
        self.exp_config = exp_config

    def on_train_begin(self, logs=None):
        if self.exp_config is not None:
            self.logger.log_dictionary("experiment_config", self.exp_config.to_dict())



class GenerativeLoggerCallback(ExperimentLogger):
    def __init__(
        self,
        log_dir: str,
        exp_config: Config = None,
        monitor_batch: tf.Tensor = None,
        monitor_update_freq: int = 0,
    ):
        super().__init__(log_dir, exp_config)
        self.monitor_batch = monitor_batch
        self.monitor_update_freq = monitor_update_freq

    def _log_image_grid(self, epoch):
        reconstructed_images = self.model.reconstruct_images(self.monitor_batch)
        synthetic_images = self.model.sample_images(num_images=self.monitor_batch.shape[0])
        
        images_to_plot = [self.monitor_batch, reconstructed_images, synthetic_images]

        titles = ['Validation Data', 'Reconstrutions', 'Synthetic Samples']
        n_subgrid_images = int(math.sqrt(self.monitor_batch.shape[0]))
        fig = plotters.nested_image_plot(
            images_to_plot,
            titles,
            image_range=(-1, 1),
            inner_rows=n_subgrid_images,
            inner_cols=n_subgrid_images
        )
        self.logger.log_figure('image_grid', plotters.plot_to_image(fig), epoch)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)

        if self.monitor_batch is not None:
            self._log_image_grid(0)

    def on_epoch_end(self, epoch, logs=None):
        is_save_epoch = epoch % self.monitor_update_freq == 0
        if self.monitor_batch is not None and is_save_epoch:
            # TODO get prediction and plot figure
            self._log_image_grid(epoch)
