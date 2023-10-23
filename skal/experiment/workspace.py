import os


class Workspace:
    def __init__(self, root_dir: str, exp_name: str = None):
        self.root_dir = root_dir
        self.exp_name = exp_name

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        if not os.path.exists(value):
            raise ValueError(f"Invalid workspace root directory {value}")
        self._root_dir = value

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value):
        if value is None:
            value = len([d for d in os.listdir(self.root_dir) 
                            if os.path.isdir(os.path.join(self.root_dir, d))])

        self._exp_name = str(value)

    @property
    def experiment_dir(self):
        return os.path.join(self.root_dir, self.exp_name)

    @property
    def logs_dir(self):
        return os.path.join(self.experiment_dir, 'logs')

    @property
    def checkpoints_dir(self):
        return os.path.join(self.experiment_dir, 'checkpoints')

    @property
    def save_dir(self):
        return os.path.join(self.experiment_dir, 'savefiles')

    def make_experiment_dirs(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Created the experiment workspace at {self.experiment_dir}")
