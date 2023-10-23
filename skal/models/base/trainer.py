from dataclasses import dataclass
from typing import Tuple


class Trainer:
    @staticmethod    
    def train_model(model, train_ds, val_ds, exp_config, workspace):
        raise NotImplementedError
