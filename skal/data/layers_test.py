import unittest
import tensorflow as tf

from skal.data import layers


class TestPreprocessing(unittest.TestCase):
    def test_random_patches(self):
        test_tensor = tf.ones(shape=[1024, 1024, 3])
        patch_extractor = layers.RandomPatches(num_patches=50, patch_size=[128, 128])
        patches = patch_extractor(test_tensor)
        self.assertEqual(patches.shape, [50, 128, 128, 3])
        

if __name__ == '__main__':
    unittest.main()
