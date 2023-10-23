import tensorflow as tf


def ssim_loss(y_true, y_pred):
    # Map [-1, 1] to [0, 1]
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0
    
    ssim_loss = 1.0 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = tf.reduce_mean(ssim_loss)
    
    return ssim_loss


def cosine_distance_loss(y_true, y_pred):
    cosine_distance = 1 - tf.keras.losses.cosine_similarity(y_true, y_pred)
    cosine_distance = tf.reduce_mean(cosine_distance)
    return cosine_distance 