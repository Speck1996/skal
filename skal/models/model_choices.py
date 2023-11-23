from enum import Enum

from skal.models.base.loader import ModelLoader
from skal.models.bigan.loader import BiganLoader
from skal.models.ganomaly.loader import GanomalyLoader
from skal.models.fanogan.loader import FanoganLoader
from skal.exceptions.model import MissingModel


class AvailableModels(Enum):
    BIGAN = "bigan"
    GANOMALY = "ganomaly"
    FANOGAN = "fanogan"


class LoaderFactory:
    @staticmethod
    def get_loader(model_type) -> ModelLoader:
        try:
            model_type = AvailableModels(model_type)
        except Exception as exc:
            raise MissingModel(str(model_type)) from exc
        
        if model_type == model_type.BIGAN:
            return BiganLoader
        elif model_type == model_type.GANOMALY:
            return GanomalyLoader
        elif model_type == model_type.FANOGAN:
            return FanoganLoader
        else:
            raise NotImplementedError
