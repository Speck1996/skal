import os
import tensorflow as tf


class BiGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        encoder,
        latent_dim=128,
        rec_weight=0.1,
        seed=None,
    ):
        super(BiGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.encoder = encoder
        self.rec_weight = rec_weight
        self.latent_dim = latent_dim
        self.seed = seed

    def compile(
        self,
        d_optimizer,
        g_optimizer,
        e_optimizer,
        d_loss_fn,
        g_loss_fn,
        e_loss_fn,
        rec_loss_fn,
        rec_metric,
    ):
        super(BiGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.e_optimizer = e_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.e_loss_fn = e_loss_fn
        self.rec_loss_fn = rec_loss_fn
        self.rec_metric = rec_metric

    @property
    def metrics(self):
        return [self.rec_metric]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Batch size
        batch_size = tf.shape(data)[0]

        # generator and encoder training cycle
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed
        )

        with tf.GradientTape() as d_tape:
            # Generate fake images from the latent vector
            fake_images = self.generator([random_latent_vectors], training=True)
            enc_noises = self.encoder([data], training=True)
            fake_outs = self.discriminator(
                [fake_images, random_latent_vectors], training=True
            )
            real_outs = self.discriminator([data, enc_noises], training=True)
            # final losses
            d_loss = self.d_loss_fn(real_outs, fake_outs)

        d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )

        # generator and encoder training cycle
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed
        )

        with tf.GradientTape(persistent=True) as ge_tape:
            # Generate fake images from the latent vector
            fake_images = self.generator([random_latent_vectors], training=True)
            enc_noises = self.encoder([data], training=True)
            rec_images = self.generator([enc_noises], training=True)
            fake_outs = self.discriminator(
                [fake_images, random_latent_vectors], training=True
            )
            real_outs = self.discriminator([data, enc_noises], training=True)
            # final losses
            g_loss = self.g_loss_fn(fake_outs)
            # reconstruction loss
            rec_loss = self.rec_loss_fn(data, rec_images)
            e_loss = self.e_loss_fn(real_outs)
            tot_e_loss = e_loss + self.rec_weight * rec_loss

        g_gradient = ge_tape.gradient(g_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )

        e_gradient = ge_tape.gradient(tot_e_loss, self.encoder.trainable_variables)

        self.e_optimizer.apply_gradients(
            zip(e_gradient, self.encoder.trainable_variables)
        )

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "e_loss": e_loss,
            "id_loss": rec_loss,
        }

    def test_step(self, data):
        enc_noises = self.encoder([data], training=False)
        recd_images = self.generator([enc_noises], training=False)
        self.rec_metric.update_state(data, recd_images)

        return {m.name: m.result() for m in self.metrics}

    def sample_images(self, num_images):
        # generator and encoder training cycle
        random_latent_vectors = tf.random.normal(
            shape=(num_images, self.latent_dim), seed=self.seed
        )
        synthetic_images = self.generator(random_latent_vectors, training=False)

        return synthetic_images

    def reconstruct_images(self, real_images):
        latent_codes = self.encoder(real_images, training=False)
        rec_images = self.generator(latent_codes, training=False)
        return rec_images
    
    def save_weights(self, weights_dir):
        self.discriminator.save_weights(os.path.join(weights_dir, 'discriminator.keras'))
        self.generator.save_weights(os.path.join(weights_dir, 'generator.keras'))
        self.encoder.save_weights(os.path.join(weights_dir, 'encoder.keras'))
    
    def load_weights(self, weights_dir):
        self.discriminator.load_weights(os.path.join(weights_dir, 'discriminator.keras'))
        self.generator.load_weights(os.path.join(weights_dir, 'generator.keras'))
        self.encoder.load_weights(os.path.join(weights_dir, 'encoder.keras'))