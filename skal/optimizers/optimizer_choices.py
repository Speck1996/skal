import tensorflow as tf
from enum import Enum, auto


class AvailableOptimizers(Enum):
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name, optimizer_args) -> tf.keras.optimizers.Optimizer:
        try:
            optimizer_type = AvailableOptimizers(optimizer_name)
        except KeyError as exc:
            raise ValueError(f"Invalid model type {optimizer_name}") from exc
        
        if optimizer_type == AvailableOptimizers.ADAM:
            return tf.keras.optimizers.Adam(**optimizer_args)
        elif optimizer_type == AvailableOptimizers.RMSPROP:
            return tf.keras.optimizers.RMSprop(**optimizer_args)
        elif optimizer_type == AvailableOptimizers.ADAMW:
            return tf.keras.optimizers.AdamW(**optimizer_args)
        else:
            raise ValueError(f"Invalid optimizer {optimizer_type}")
