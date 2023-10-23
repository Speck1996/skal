import tensorflow as tf
from enum import Enum, auto

from skal.models.base import loader
from skal.models.bigan.loader import BiganLoader

class AvailableModels(Enum):
    BIGAN = "bigan"


class LoaderFactory:
    @staticmethod
    def get_loader(model_name) -> loader.ModelLoader:
        try:
            model_architecture = AvailableModels(model_name)
        except KeyError as exc:
            raise ValueError(f"Invalid model type {model_name}") from exc
        
        if model_architecture == AvailableModels.BIGAN:
            return BiganLoader()
        else:
            raise ValueError(f"Unknown model {model_name}")
