import tensorflow as tf

from skal.utils.utils import pretty_json

class TensorboardLogger:
    def __init__(self, logdir) -> None:
        self.writer = tf.summary.create_file_writer(
            logdir=logdir
        )

    def log_dictionary(self, name, dictionary, step=0) -> None:
        with self.writer.as_default():
            tf.summary.text(name, pretty_json(dictionary), step=step)

    def log_text(self, name, text, step=0) -> None:
        with self.writer.as_default():
            tf.summary.text(name, text, step=step)

    def log_scalar(self, name, scalar, step=0) -> None:
        with self.writer.as_default():
            tf.summary.scalar(name, scalar, step=step)

    def log_figure(self, name, figure, step=0) -> None:
        # Convert to image and log
        with self.writer.as_default():
            tf.summary.image(name, figure, step=step)
