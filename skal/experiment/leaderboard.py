import pandas as pd
import os


class Leaderboard:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            self.data = pd.DataFrame(
                columns=["Experiment ID", "Date", "Model Name", "AUROC", "AUPR"]
            )
            self.save_to_file()
        else:
            self.data = pd.read_csv(file_path)

    def add_entry(self, experiment_id, date, model_name, auroc, aupr):
        new_entry = {
            "Experiment ID": experiment_id,
            "Date": date,
            "Model Name": model_name,
            "AUROC": auroc,
            "AUPR": aupr,
        }
        self.data = self.data.append(new_entry, ignore_index=True)
        self.save_to_file()

    def get_leaderboard(self):
        return self.data

    def save_to_file(self):
        self.data.to_csv(self.file_path, index=False)
