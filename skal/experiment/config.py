import yaml
import uuid
import petname
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Config:
    seed: int
    epochs: int
    batch_size: int
    val_split: float
    count: int
    # NOTE the following attributes could be potentially
    # refactored to other dataclasses
    preprocessor: dict
    augmenter: dict
    model: dict
    detector: dict
    name: str = None
    datetime: str = None
    uuid: str = None

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())[:8]

        if self.datetime is None:
            self.datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.name is None:
            self.name = petname.Generate(words=2, separator="-")

    def to_dict(self):
        return asdict(self)

