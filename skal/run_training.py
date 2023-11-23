import click
import os

from skal.data.folders import FolderFactory, AvailableFolders
from skal.data.preprocessors import PreprocessorBuilder
from skal.data.augmenters import AugmenterBuilder
from skal.data.dataset_builder import AnomalyDatasetBuilder
from skal.models.model_choices import LoaderFactory, AvailableModels
from skal.experiment.config import Config
from skal.utils import utils
from skal.experiment.workspace import Workspace
from skal.utils.utils import set_gpu


@click.command()
@click.option('--training-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--dataset-format', type=click.Choice(list([member.value for member in AvailableFolders])), prompt='Choose the proper dataset format', help='Select dataset format', default='mvtec')
@click.option('--model', type=click.Choice(list([member.value for member in AvailableModels])), prompt='Choose a model', help='Select a model')
@click.option('--config-path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--experiment-dir', type=click.Path(exists=False, file_okay=False, dir_okay=True))
def run_training(training_dir, dataset_format, model, config_path, experiment_dir):
    """
    Trains a deep learning generative anomaly detection model on a given dataset.

    Args:
        training_dir (str): The directory path of the training data.
        dataset_format (str): The format of the dataset (e.g., 'mvtec', 'folder').
        model (str): The type of model to be trained (e.g., 'bigan', 'fanogan').
        config_path (str): The path to the configuration file.
        experiment_dir (str): The directory to save the experiment results.

    Returns:
        None. The function performs the training and saves the model weights, but does not return any output.
    """
    set_gpu()
    click.echo(f"{training_dir} with {model}.")
    exp_params = utils.load_yaml_file(config_path)
    config = Config(**exp_params)
    exp_ws = Workspace(root_dir=experiment_dir)
    folder = FolderFactory.get_folder(dataset_format, training_dir)

    training_paths = folder.get_training_paths(shuffle=True, seed=config.seed)
    print(f"Found {len(training_paths)} training paths")

    preprocessor = PreprocessorBuilder.get_preprocessor(config.preprocessor)
    augmenter = AugmenterBuilder.augmenter_from_config(config.augmenter)
    dataset_builder = AnomalyDatasetBuilder(
        folder, preprocessor, augmenter=augmenter, seed=config.seed
    )
    train_ds, val_ds = dataset_builder.train_val_ds_from_folder(
        shuffle=True, batch_size=config.batch_size, val_split=config.val_split, ds_count=config.count)

    loader = LoaderFactory.get_loader(config.model['name'])
    model = loader.load_model_from_config(config.model, seed=config.seed)
    trainer = loader.load_trainer()
    exp_ws.make_experiment_dirs()

    print("Everything is ready. Starting training...")
    trainer.train_model(model, train_ds, val_ds, config, exp_ws)
    model.save_weights(exp_ws.save_dir)
    print("Job done!")


if __name__ == "__main__":
    run_training()
