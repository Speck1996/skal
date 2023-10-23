import click
import os
from tqdm import tqdm

from skal.data.folders import FolderFactory, AvailableFolders
from skal.data.dataset_builder import AnomalyDatasetBuilder
from skal.models.model_choices import LoaderFactory, AvailableModels
from skal.experiment.config import Config
from skal.utils import utils
from skal.utils.utils import set_gpu
from skal.anomaly_map import ops


@click.command()
@click.option('--test-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--dataset-format', type=click.Choice(list([member.value for member in AvailableFolders])), 
              prompt='Choose the proper dataset format', help='Select dataset format', default='mvtec')
@click.option('--model', type=click.Choice(list([member.value for member in AvailableModels])),
              prompt='Choose a model', help='Select a model')
@click.option('--weights-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--config-path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--experiment-dir', type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option('--dest-dir', type=click.Path(exists=False, file_okay=False, dir_okay=True))
def run_inference(test_dir, dataset_format, model, weights_dir, config_path, experiment_dir, dest_dir):
    set_gpu()
    click.echo(f"{test_dir} with {model}.")
    exp_params = utils.load_yaml_file(config_path)
    config = Config(**exp_params)
    folder = FolderFactory.get_folder(dataset_format, test_dir)

    folder_test_paths = folder.get_test_paths()
    print(f"Found {len(folder_test_paths)} test paths")
    loader = LoaderFactory.get_loader(config.model['name'])
    detector = loader.load_detector(config.detector, weights_dir)
    dataset_builder = AnomalyDatasetBuilder(
        folder, preprocessor=None, augmenter=None, seed=config.seed
    )
    test_ds = dataset_builder.get_test_ds(batch_size=1)

    anomaly_map_folder = os.path.join(dest_dir, "pred")
    os.makedirs(anomaly_map_folder, exist_ok=True)
    for (test_img, _), (img_path, _) in tqdm(zip(test_ds, folder_test_paths)):
        pred = detector.score_anomalies(test_img).numpy()
        anomaly_image = ops.anomaly_map_to_img(pred)
        test_label = img_path.split(os.sep)[-2] # replace with dirname of test image path
        image_name = img_path.split(os.sep)[-1].split(".")[0]
        os.makedirs(os.path.join(anomaly_map_folder, test_label), exist_ok=True)
        image_path = os.path.join(anomaly_map_folder, test_label, image_name + ".tiff")
        anomaly_image.save(image_path)

    print("All done!")


if __name__ == "__main__":
    run_inference()
