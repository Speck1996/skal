import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow_io as tfio
from functools import partial
import os



@tf.function
def decode_tiff(image: tf.Tensor):
    # special procedure needed for tiff images since tf 2.8 decode image
    # does not handle .tiff/tif images
    image = tfio.experimental.image.decode_tiff(image)
    image = tf.ensure_shape(image, [None, None, 4])
    image = tfio.experimental.color.rgba_to_rgb(image)
    image = tf.image.rgb_to_grayscale(image)

    return image


@tf.function
def decode_image(image_path: tf.Tensor):
    image_format = tf.strings.split(image_path, sep=".")[-1]

    is_jpeg = tf.equal(image_format, tf.constant("jpg")) | tf.equal(
        image_format, tf.constant("jpeg")
    )
    is_png = tf.equal(image_format, tf.constant("png"))
    is_tiff = tf.equal(image_format, tf.constant("tiff")) | tf.equal(
        image_format, tf.constant("tif")
    )

    image_bytes = tf.io.read_file(image_path)
    if tf.equal(is_png, tf.constant(True)):
        return tf.io.decode_png(image_bytes)
    elif tf.equal(is_jpeg, tf.constant(True)):
        return tf.io.decode_jpeg(image_bytes)
    elif tf.equal(is_tiff, tf.constant(True)):
        return decode_tiff(image_bytes)
    else:
        return tf.io.decode_image(image_bytes, expand_animations=False)


@tf.function
def decode_gt(gt_path: tf.Tensor, img: tf.Tensor):
    # Placeholder for the path

    # Check if the path exists using os.path.exists
    if  tf.strings.regex_full_match(gt_path, ".*good.*"):
        # special case for the mvtec test folder, which does not contain
        # ground truth images for `good` sample, since they are just
        # black images.
        return tf.zeros(shape=tf.shape(img), dtype=tf.uint8)
    else:
        gt = tf.io.read_file(gt_path)
        gt = tf.io.decode_image(gt, expand_animations=False)
        return gt


@tf.function
def min_max_normalization(image):
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image


@tf.function
def tile_image(img: tf.Tensor, patch_size: tuple):
    target_height = tf.shape(img)[0] // patch_size[0] * patch_size[0]
    target_width = tf.shape(img)[1] // patch_size[1] * patch_size[1]

    img = tf.image.resize_with_crop_or_pad(
        img, target_height=target_height, target_width=target_width
    )

    if len(tf.shape(img)) == 3:
        img = tf.expand_dims(img, axis=0)

    patches = tf.image.extract_patches(
        img, [1, *patch_size, 1], [1, *patch_size, 1], [1, 1, 1, 1], padding="VALID"
    )
    patches = tf.reshape(patches, shape=(-1, *patch_size, tf.shape(img)[-1]))

    return patches


@tf.function
def untile_image(patches_tensor, image_shape):
    num_patches = patches_tensor.shape[0]
    image_channels = patches_tensor.shape[-1]
    patch_size = patches_tensor.shape[1:3]
    target_height = image_shape[0] // patch_size[0] * patch_size[0]
    target_width = image_shape[1] // patch_size[1] * patch_size[1]
    # Reshape the tensor to [1, num_patches * patch_size, patch_size, image_channels]
    reshaped_tensor = tf.reshape(
        patches_tensor, [1, num_patches * patch_size[0], patch_size[1], image_channels]
    )

    # Transpose dimensions to [1, patch_size, num_patches * patch_size, image_channels]
    transposed_tensor = tf.transpose(reshaped_tensor, [0, 2, 1, 3])

    # Reshape again to [1, image_shape, image_shape, image_channels]
    final_image_tensor = tf.reshape(
        transposed_tensor, [1, image_shape[0], image_shape[1], image_channels]
    )
    final_image_tensor = tf.image.resize_with_crop_or_pad(
        final_image_tensor, target_height, target_width
    )

    return final_image_tensor
