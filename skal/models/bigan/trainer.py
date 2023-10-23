import tensorflow as tf

from skal.models.base.trainer import Trainer
from skal.optimizers.optimizer_choices import OptimizerFactory
from skal.losses import adversarial
from skal.callbacks.logger import GenerativeLoggerCallback


class BiganTrainer(Trainer):
    @staticmethod
    def train_model(model, train_ds, val_ds, exp_config, workspace):
        epochs = exp_config.epochs
        
        # loading optimizers and loss functions
        g_optimizer_name = exp_config.model['generator']['optimizer_name']
        g_optimizer_kwargs = exp_config.model['generator']['optimizer_args']
        g_optimizer = OptimizerFactory.get_optimizer(g_optimizer_name, g_optimizer_kwargs)
        g_loss = adversarial.generator_ls_loss

        d_optimizer_name = exp_config.model['discriminator']['optimizer_name']
        d_optimizer_kwargs = exp_config.model['discriminator']['optimizer_args']
        d_optimizer = OptimizerFactory.get_optimizer(d_optimizer_name, d_optimizer_kwargs)
        d_loss = adversarial.discriminator_ls_loss

        e_optimizer_name = exp_config.model['encoder']['optimizer_name']
        e_optimizer_kwargs = exp_config.model['encoder']['optimizer_args']
        e_optimizer = OptimizerFactory.get_optimizer(e_optimizer_name, e_optimizer_kwargs)
        e_loss = adversarial.encoder_ls_loss
        #rec_loss = reconstruction.ssim_loss
        rec_loss = tf.keras.losses.MeanAbsoluteError()
        validation_metric = tf.keras.metrics.MeanAbsoluteError()
        
        # callbacks
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", restore_best_weights=True, patience=10, start_from_epoch=50)
        #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(workspace.checkpoints_dir, monitor="val_loss")
        tensorboard_cb = tf.keras.callbacks.TensorBoard(workspace.logs_dir, histogram_freq=5)
        validation_batch = next(iter(val_ds))
        exp_logger_cb = GenerativeLoggerCallback(workspace.logs_dir, exp_config, validation_batch, monitor_update_freq=5)
        cbks = [early_stop_cb, tensorboard_cb, exp_logger_cb]
        
        model.compile(d_optimizer, g_optimizer, e_optimizer, d_loss, g_loss, e_loss, rec_loss, validation_metric)
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbks)
        
        return model
