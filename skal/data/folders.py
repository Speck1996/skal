from typing import List, Tuple
from abc import abstractmethod
from pathlib import Path
import random
from enum import Enum

from skal.exceptions.data import EmptyFolderError


class AnomalyFolder:
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)

    @abstractmethod
    def get_training_paths(self, shuffle: bool = True) -> List[str]:
        """ Method that extracts all the file paths from the root directory.

        Args:
            shuffle (bool): If true shuffle the paths before returning them,
            otherwise they are returned by alphabetical order.

        Raises:
            NotImplementedError: 

        Returns:
            List[str]: List of all the file paths.
        """
        raise NotImplementedError

    @abstractmethod
    def get_test_paths(self) -> List[Tuple[str, str]]:
        raise NotImplementedError


class SimpleFolder(AnomalyFolder):
    def __init__(self, data_folder: str):
        super().__init__(data_folder)


    def get_training_paths(self, shuffle: bool = True, seed: int = None) -> List[str]:
        """ Method that extracts all the file paths from the root directory.

        Args:
            shuffle (bool): If true shuffle the paths before returning them,
            otherwise they are returned by alphabetical order.

        Raises:
            NotImplementedError: 

        Returns:
            List[str]: List of all the file paths.
        """
        training_dir = self.data_folder / "train"
        image_paths = [str(path) for path in training_dir.glob("*.png")]
        image_paths = image_paths + [str(path) for path in training_dir.glob("*.tiff")]
        image_paths = image_paths + [str(path) for path in training_dir.glob("*.jpg")]

        if len(image_paths) == 0:
            raise EmptyFolderError(training_dir)

        if shuffle:
            random.seed(seed)
            random.shuffle(image_paths)

        return image_paths

    def get_test_paths(self) -> List[Tuple[str, str]]:
        test_images_dir = self.data_folder / "test"
        test_gt_dir = self.data_folder / "ground_truth"

        image_paths = sorted([str(path) for path in test_images_dir.rglob("*.png")])
        image_paths = image_paths + [str(path) for path in test_images_dir.glob("*.tiff")]
        image_paths = image_paths + [str(path) for path in test_images_dir.glob("*.jpg")]
        gt_paths = sorted([str(path) for path in test_gt_dir.rglob("*.png")])
        gt_paths = gt_paths + sorted([str(path) for path in test_gt_dir.rglob("*.tiff")])
        gt_paths = gt_paths + sorted([str(path) for path in test_gt_dir.rglob("*.jpg")])

        # adding mockup names for the ground truth of good images
        if len(image_paths) != len(gt_paths):
            raise ValueError("Mismatch between number of gt images and test images")

        return list(zip(image_paths, gt_paths))


class MvtecFolder(AnomalyFolder):
    def get_training_paths(self, shuffle: bool = True, seed: int = None) -> List[str]:
        """ Method that extracts all the file paths from the root directory.

        Args:
            shuffle (bool): If true shuffle the paths before returning them,
            otherwise they are returned by alphabetical order.

        Raises:
            NotImplementedError: 

        Returns:
            List[str]: List of all the file paths.
        """
        training_dir = self.data_folder / "train" / "good"
        image_paths = [str(path) for path in training_dir.glob("*.png")]

        if len(image_paths) == 0:
            raise EmptyFolderError(training_dir)

        if shuffle:
            random.seed(seed)
            random.shuffle(image_paths)

        return image_paths

    def get_test_paths(self) -> List[Tuple[str, str]]:
        test_images_dir = self.data_folder / "test"
        test_gt_dir = self.data_folder / "ground_truth"

        image_paths = sorted([path for path in test_images_dir.rglob("*.png")])
        gt_paths = sorted([test_gt_dir / path.parent / path.name for path in image_paths])
        image_paths = [str(path) for path in image_paths]
        gt_paths = [str(path) for path in gt_paths]
        # adding mockup names for the ground truth of good images
        if len(image_paths) != len(gt_paths):
            raise ValueError("Mismatch between number of gt images and test images")

        return list(zip(image_paths, gt_paths))
 
 
class NanotwiceFolder(AnomalyFolder):
    def __init__(self, data_folder: str):
        super().__init__(data_folder)

    def get_training_paths(self, shuffle: bool = True, seed: int = None) -> List[str]:
        """ Method that extracts all the file paths from the root directory.

        Args:
            shuffle (bool): If true shuffle the paths before returning them,
            otherwise they are returned by alphabetical order.

        Raises:
            NotImplementedError: 

        Returns:
            List[str]: List of all the file paths.
        """
        training_dir = self.data_folder / "Normal"
        image_paths = [str(path) for path in training_dir.glob("*.tif")]

        if len(image_paths) == 0:
            raise EmptyFolderError(training_dir)

        if shuffle:
            random.seed(seed)
            random.shuffle(image_paths)

        return image_paths

    def get_test_paths(self) -> List[Tuple[str, str]]:
        test_images_dir = self.data_folder / "Anomalous" / "images"
        test_gt_dir = self.data_folder / "Anomalous" / "gt"

        image_paths = sorted([str(path) for path in test_images_dir.rglob("*.tif")])
        gt_paths = sorted([str(path) for path in test_gt_dir.rglob("*.png")])
        # adding mockup names for the ground truth of good images
        if len(image_paths) != len(gt_paths):
            raise ValueError("Mismatch between number of gt images and test images")

        return list(zip(image_paths, gt_paths))
    
    
class AvailableFolders(Enum):
    NANOTWICE = "nanotwice"
    MVTECAD = "mvtecad"
    SIMPLE = "simple"
    
    
class FolderFactory:
    @staticmethod
    def get_folder(folder_format, folder_root):
        try:
            selected_folder = AvailableFolders(folder_format)
        except KeyError as exc:
            raise ValueError(f"Invalid folder type {folder_format}") from exc
        
        if selected_folder == AvailableFolders.NANOTWICE:
            return NanotwiceFolder(folder_root)
        elif selected_folder == AvailableFolders.MVTECAD:
            return MvtecFolder(folder_root)
        elif selected_folder == AvailableFolders.SIMPLE:
            return AnomalyFolder(folder_root)