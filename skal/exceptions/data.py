class EmptyFolderError(Exception):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.message = f"The folder '{folder_path}' is empty."

    def __str__(self):
        return self.message


class MissingGroundTruthError(Exception):
    def __init__(self, image_path):
        self.image_path = image_path
        self.message = f"The image '{image_path}' has no corresponding ground truth."

    def __str__(self):
        return self.message
