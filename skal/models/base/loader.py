class ModelLoader:
    @staticmethod
    def load_model_from_config(config, weights_dir=None):
        raise NotImplementedError

    @staticmethod
    def load_trainer():
        raise NotImplementedError

    @staticmethod
    def load_detector(config, weights_dir=None):
        raise NotImplementedError
