import os
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
import json
import yaml
import re
import uuid

from typing import Dict, Any


def load_yaml_file(file_path: str) -> Dict[Any, str]:
    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content


def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))


def create_dir(path):
    if not os.path.exists(path):
        print(f"Creating directory {path}")
        os.mkdir(path)
    else:
        print(f"Directory {path} already exists")


def set_gpu():
    os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
    #tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # TODO precision variable
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
