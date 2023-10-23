import os
from tensorflow import keras

class ModelSaverCallback(keras.callbacks.Callback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def on_train_end(self, logs=None):
        save_path = os.path.join(self.save_dir, self.model.name + ".keras")
        self.model.save_weights(save_path)
