import unittest
import tensorflow as tf

from skal.ops import images_tfops


class TestTiling(unittest.TestCase):
    def test_tiling_patches(self):
        # Test tiling
        test_tensor = tf.ones(shape=[1, 1024, 1024, 3])
        tiled_patches = images_tfops.tile_image(test_tensor, [256, 256])
        self.assertEqual(tiled_patches.shape, [16, 256, 256, 3])

    def test_untiling(self):
        test_tensor = tf.ones(shape=[16, 256, 256, 3])
        original_image = images_tfops.untile_image(test_tensor, [1024, 1024])
        self.assertEqual(original_image.shape, [1, 1024, 1024, 3])


class TestStringSplit(unittest.TestCase):
    def test_string_split(self):
        test_tensor = tf.Variable("mockup_path.png")
        format_tensor = tf.strings.split(test_tensor, sep=".")[-1]
        self.assertEqual(format_tensor, "png")


if __name__ == "__main__":
    unittest.main()
