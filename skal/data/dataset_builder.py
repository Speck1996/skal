import math
import os
from typing import Tuple
import tensorflow as tf
from typing import List

from skal.data.folders import AnomalyFolder
from skal.ops.images_tfops import decode_image, decode_gt


class DatasetBuilder:
    def __init__(
        self,
        folder: AnomalyFolder,
        preprocessor: tf.keras.Sequential,
        augmenter: tf.keras.Sequential,
        seed: int = None,
    ):
        self.folder = folder
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.seed = seed

    def train_val_ds_from_folder(self, shuffle: bool, batch_size: int, val_split:float = 0.2):
        raise NotImplementedError

    def test_ds_from_folder(self, shuffle: bool, batch_size: int):
        raise NotImplementedError


class AnomalyDatasetBuilder(DatasetBuilder):
    
    def train_val_ds_from_folder(
        self, shuffle: bool, batch_size: int, val_split:float = 0.2, ds_count: int = 1
    ) -> tf.data.Dataset:
        training_paths = self.folder.get_training_paths(shuffle=shuffle, seed=self.seed)
        
        if val_split >= 1.0:
            raise ValueError("Expected val_split to be in the range [0., 1.)")
        
        if val_split > 0.0 and val_split < 1.0:
            val_paths = training_paths[:int(len(training_paths) * val_split)]
            training_paths = training_paths[int(len(training_paths) * val_split):]
        else:
            # empty validation set
            val_paths = []

        train_ds = tf.data.Dataset.from_tensor_slices(training_paths)
        train_ds = train_ds.map(decode_image, tf.data.AUTOTUNE)
        
        if ds_count > 1:
            train_ds = train_ds.repeat(ds_count)
        
        if self.preprocessor:
            train_ds = train_ds.map(self.preprocessor, tf.data.AUTOTUNE)
        
        train_ds = train_ds.cache()
        
        if shuffle:
            train_ds = train_ds.shuffle(batch_size * 10, seed=self.seed)

        if self.augmenter:
            train_ds = train_ds.map(self.augmenter, tf.data.AUTOTUNE)

        if len(train_ds.element_spec.shape) == 3:
            train_ds = train_ds.batch(batch_size)
        else:
            train_ds = train_ds.rebatch(batch_size)

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # join train and val_ds creation methods
        val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
        val_ds = val_ds.map(decode_image, tf.data.AUTOTUNE)
        val_ds = val_ds.repeat(2500)
        if self.preprocessor:
            val_ds = val_ds.map(self.preprocessor, tf.data.AUTOTUNE)
        val_ds = val_ds.cache()
        if len(val_ds.element_spec.shape) == 3:
            val_ds = val_ds.batch(batch_size)
        else:
            val_ds = val_ds.rebatch(batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def get_test_ds(self, batch_size: int = 32) -> tf.data.Dataset:
        test_gt_paths = self.folder.get_test_paths()

        test_ds = tf.data.Dataset.from_tensor_slices(test_gt_paths)

        test_ds = test_ds.map(lambda paths: (decode_image(paths[0]), paths[1]), tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda img, gt_paths: (img, decode_gt(gt_paths, tf.shape(img))), tf.data.AUTOTUNE)

        if self.preprocessor:
            test_ds = test_ds.map(
                lambda img, gt: (self.preprocessor(img), self.preprocessor(gt))
            )

        test_ds = test_ds.cache()
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        return test_ds
