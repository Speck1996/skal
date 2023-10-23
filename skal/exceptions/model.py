class MissingModel(Exception):
    def __init__(self, model_name):
        self.model_name = model_name
        self.message = f"Unavailable model '{model_name}'."

    def __str__(self):
        return self.message
