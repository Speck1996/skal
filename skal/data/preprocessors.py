import tensorflow as tf

from skal.data import layers


class PreprocessorBuilder:
    @staticmethod
    def get_preprocessor(
        preprocessor_config: dict = {}, seed: int = None
    ) -> tf.keras.Sequential:
        
        if len(preprocessor_config) == 0:
            print("Skipping preprocessor creation. Empty Configuration!")
            return

        preprocessor = tf.keras.Sequential()

        # NOTE not really proud of this part of code probably preprocessors and augmenters
        # are things that should be refactored.
        # Things I am considering: strategy pattern with predefined augmenters
        # and ability to add custom augmenters or just a base customizable builder for both.
        # I don't want to overcomplicate things right now, so you get the good'ol for + if elif loop
        # and some copy pasted code :)
        for layer in preprocessor_config:
            layer_params = preprocessor_config[layer]

            if layer == "resizing":
                preprocessor.add(tf.keras.layers.Resizing(**layer_params))
            elif layer == "rescaling":
                preprocessor.add(tf.keras.layers.Rescaling(**layer_params))
            elif layer == "tiling":
                preprocessor.add(layers.Tiling(**layer_params))
            elif layer == "patches":
                preprocessor.add(layers.RandomPatches(**layer_params, seed=seed))
            elif layer == "resizepadcrop":
                preprocessor.add(layers.ResizePadCrop(**layer_params))
            else:
                raise NotImplementedError

        return preprocessor
