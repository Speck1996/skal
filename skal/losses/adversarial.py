"""Loss functions."""
import tensorflow as tf
from tensorflow import keras

bce = keras.losses.BinaryCrossentropy(from_logits=True)
mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()


def discriminator_ls_loss(real_output, fake_output):
    real_loss = mse(tf.ones_like(real_output), real_output)
    fake_loss = mse(tf.zeros_like(fake_output), fake_output)

    return (fake_loss + real_loss) * 0.5


def generator_ls_loss(fake_output):
    fake_loss = mse(tf.ones_like(fake_output), fake_output)

    return fake_loss


def encoder_ls_loss(real_output):
    real_loss = mse(tf.zeros_like(real_output), real_output)

    return real_loss


def discriminator_wass_loss(real_output, fake_output, gp_weight = 10, grads = None):
    real_loss = tf.reduce_mean(real_output, axis=-1)
    fake_loss = tf.reduce_mean(fake_output, axis=-1)
    total_loss = fake_loss - real_loss
    
    if grads:
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        total_loss = total_loss + gp_weight * gp
    
    return total_loss


def generator_wass_loss(fake_output):
    return - tf.reduce_mean(fake_output)


def encoder_wass_loss(real_ouput):
    return tf.reduce_mean(real_ouput)