from tensorflow import keras
import tensorflow_addons as tfa

from skal.models.base.loader import ModelLoader
from skal.models.blocks import ConvBlock
from skal.models.layers import DepthToSpace, ReflectionPadding2D, DropConcatenate
from skal.models.bigan.model import BiGAN
from skal.models.bigan.trainer import BiganTrainer
from skal.models.bigan.detector import BiganDetector
from skal.data import preprocessors
from skal.anomaly_map.postprocessor import PostprocessorBuilder


class BiganLoader(ModelLoader):
    @staticmethod
    def _build_discriminator(input_shape, block_filters, output_shape, base_kernel_size, weights_path=None):
        kernel_initizializer = keras.initializers.RandomNormal(stddev=0.02)
        img_shape, latent_shape = input_shape
        downsampling_factor = 2 ** len(block_filters)

        img_input = keras.layers.Input(img_shape)
        latent_input = keras.layers.Input(latent_shape)
        
        if len(latent_shape) == 1:
            latent_layer = keras.layers.Reshape((1, 1, latent_shape[-1]))(
                latent_input
            )
        else:
            latent_layer = latent_input
            
        # Adapter block to project the latent code
        # to the image dimension
        z = keras.layers.Conv2DTranspose(
            1,
            base_kernel_size,
            use_bias=False,
            padding="valid",
            kernel_initializer='orthogonal',
            activation=None
        )(latent_layer)
        z = keras.layers.Conv2D(
            downsampling_factor**2,
            1,
            use_bias=False,
            padding="valid",
            kernel_initializer='orthogonal',
            activation=None
        )(z)
        z = keras.layers.SpatialDropout2D(0.5)(z)
        z = DepthToSpace(downsampling_factor)(z)

        x = img_input
        x = keras.layers.Concatenate()([x, z])

        dilation_rate = (1, 1)
        for i, filters in enumerate(block_filters):

            x = ConvBlock(
                filters,
                padding="same",
                apply_normalization=False,
                kernel_initializer=kernel_initizializer,
                activation=keras.layers.LeakyReLU(0.2),
                drop_rate=0.00,
                dilation_rate=dilation_rate)(x)
            x = ConvBlock(
                filters,
                padding="same",
                apply_normalization=False,
                kernel_initializer=kernel_initizializer,
                activation=keras.layers.LeakyReLU(0.2),
                drop_rate=0.2,
                dilation_rate=dilation_rate)(x)

            if len(output_shape) == 1:
                x = keras.layers.AveragePooling2D((2, 2))(x)
            else:
                dilation_rate = [rate * 2 for rate in dilation_rate]
                
        x = ConvBlock(
            1 ,
            kernel_size=base_kernel_size,
            dilation_rate=dilation_rate,
            padding='valid',
            apply_normalization=False,
            kernel_initializer=kernel_initizializer,
            drop_rate=0.2
        )(x)

        if len(output_shape) == 1:
            x = keras.layers.Reshape((output_shape[0],))(x)

        discriminator = keras.models.Model(
            inputs=[img_input, latent_input], outputs=x, name="Discriminator"
        )

        if weights_path is not None:
            discriminator.load_weights(weights_path)
        
        return discriminator

    @staticmethod
    def _build_generator(input_shape, block_filters, output_shape, base_kernel_size, weights_path=None):
        #base_kernel_dim = output_shape[1] // (2**len(block_filters))
        #base_kernel_dim = 32 // (2**len(block_filters))
        kernel_initizializer = keras.initializers.RandomNormal(stddev=0.02)

        input_layer = keras.Input(shape=input_shape)

        if len(input_shape) == 1:
            x = keras.layers.Reshape((1, 1, input_shape[-1]))(input_layer)
        else:
            x = input_layer

        x = keras.layers.Conv2DTranspose(
            block_filters[0],
            base_kernel_size,
            kernel_initializer=kernel_initizializer,
            use_bias=False
        )(x)
        x = keras.layers.GroupNormalization(groups=16)(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.SpatialDropout2D(0.2)(x)
        
        for filters in block_filters:
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = ConvBlock(filters,
                          use_bias=False,
                          padding="same",
                          activation=keras.layers.LeakyReLU(0.2),
                          drop_rate=0.0,
                          kernel_initializer=kernel_initizializer)(x)
            x = ConvBlock(filters,
                          use_bias=False,
                          padding="same",
                          drop_rate=0.2,
                          activation=keras.layers.LeakyReLU(0.2),
                          kernel_initializer=kernel_initizializer)(x)

        x = ReflectionPadding2D((2, 2))(x)
        x = keras.layers.Conv2D(output_shape[-1],
                                (5, 5),
                                kernel_initializer=kernel_initizializer,
                                use_bias=False,
                                padding="valid",
                                activation="tanh")(x)

        generator = keras.models.Model(input_layer, x, name="Generator")
        
        if weights_path is not None:
            generator.load_weights(weights_path)

        return generator

    @staticmethod
    def _build_encoder(input_shape, block_filters, output_shape, base_kernel_size, weights_path=None):
        kernel_initizializer = keras.initializers.RandomNormal(stddev=0.02)

        input_layer = keras.Input(shape=input_shape)

        x = input_layer
        for filters in block_filters:
            x = ConvBlock(filters,
                          use_bias=False,
                          padding="same",
                          apply_normalization=not filters == block_filters[0],
                          activation=keras.layers.LeakyReLU(0.2),
                          drop_rate=0.00,
                          kernel_initializer=kernel_initizializer)(x)
            x = ConvBlock(filters,
                          use_bias=False,
                          padding="same",
                          drop_rate=0.2,
                          activation=keras.layers.LeakyReLU(0.2),
                          kernel_initializer=kernel_initizializer)(x)
            x = keras.layers.AveragePooling2D((2, 2))(x)

        x = ConvBlock(block_filters[-1],
                      kernel_size=base_kernel_size,
                      use_bias=False,
                      padding="valid",
                      activation=keras.layers.LeakyReLU(0.2),
                      kernel_initializer=kernel_initizializer,
                      drop_rate=0.00)(x)

        x = keras.layers.Conv2D(
            output_shape[-1],
            1,
            kernel_initializer=kernel_initizializer,
            use_bias=False,
            padding="valid",
            activation=None
        )(x)

        if len(output_shape) == 1:
            x = keras.layers.Reshape((output_shape[0],))(x)

        encoder = keras.models.Model(input_layer, x, name="Encoder")

        if weights_path is not None:
            encoder.load_weights(weights_path)

        return encoder

    @staticmethod
    def load_model_from_config(config, weights_dir=None, seed=None):
        # TODO add weights path
        generator_cfg = config['generator']
        g_input_shape = generator_cfg['input_shape']
        g_block_filters = generator_cfg['block_filters']
        g_output_shape = generator_cfg['output_shape']
        g_kernel_size = generator_cfg['base_kernel_size']
        generator = BiganLoader._build_generator(g_input_shape, g_block_filters, g_output_shape, g_kernel_size)

        # discriminator
        discriminator_cfg = config['discriminator']
        d_input_shape = discriminator_cfg['input_shape']
        d_block_filters = discriminator_cfg['block_filters']
        d_output_shape = discriminator_cfg['output_shape']
        d_kernel_size = discriminator_cfg['base_kernel_size']
        discriminator = BiganLoader._build_discriminator(d_input_shape, d_block_filters, d_output_shape, d_kernel_size)

        # encoder
        encoder_cfg = config['encoder']
        e_input_shape = encoder_cfg['input_shape']
        e_block_filters = encoder_cfg['block_filters']
        e_output_shape = encoder_cfg['output_shape']
        e_kernel_size = encoder_cfg['base_kernel_size']
        encoder = BiganLoader._build_encoder(e_input_shape, e_block_filters, e_output_shape, e_kernel_size)

        return BiGAN(discriminator, generator, encoder, seed=seed)

    @staticmethod
    def load_trainer():
        return BiganTrainer

    @staticmethod
    def load_detector(config, weights_dir=None, seed:int = None):
        preprocessor = preprocessors.PreprocessorBuilder.get_preprocessor(config['preprocessor'])
        postprocessor = PostprocessorBuilder.get_postprocessor(config['postprocessor'])
        
        # TODO add weights path
        generator_cfg = config['generator']
        g_input_shape = generator_cfg['input_shape']
        g_block_filters = generator_cfg['block_filters']
        g_output_shape = generator_cfg['output_shape']
        g_kernel_size = generator_cfg['base_kernel_size']
        generator = BiganLoader._build_generator(g_input_shape, g_block_filters, g_output_shape, g_kernel_size)

        # discriminator
        discriminator_cfg = config['discriminator']
        d_input_shape = discriminator_cfg['input_shape']
        d_block_filters = discriminator_cfg['block_filters']
        d_output_shape = discriminator_cfg['output_shape']
        d_kernel_size = discriminator_cfg['base_kernel_size']
        discriminator = BiganLoader._build_discriminator(d_input_shape, d_block_filters, d_output_shape, d_kernel_size)

        # encoder
        encoder_cfg = config['encoder']
        e_input_shape = encoder_cfg['input_shape']
        e_block_filters = encoder_cfg['block_filters']
        e_output_shape = encoder_cfg['output_shape']
        e_kernel_size = encoder_cfg['base_kernel_size']
        encoder = BiganLoader._build_encoder(e_input_shape, e_block_filters, e_output_shape, e_kernel_size)

        model = BiGAN(discriminator, generator, encoder, seed=seed)
        
        if weights_dir:
            model.load_weights(weights_dir)

        detector = BiganDetector(preprocessor, model, postprocessor)

        return detector
