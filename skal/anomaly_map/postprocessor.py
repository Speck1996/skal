import tensorflow as tf

from skal.data import layers

class PostprocessorBuilder:
    @staticmethod
    def get_postprocessor(
        postprocessor_config: dict = {}, seed: int = None
    ) -> tf.keras.Sequential:
        
        if len(postprocessor_config) == 0:
            print("Skipping postprocessor creation. Empty Configuration!")
            return

        postprocessor = tf.keras.Sequential()

        # NOTE not really proud of this part of code probably postprocessors and augmenters
        # are things that should be refactored.
        # Things I am considering: strategy pattern with predefined augmenters
        # and ability to add custom augmenters or just a base customizable builder for both.
        # I don't want to overcomplicate things right now, so you get the good'ol for + if elif loop
        # and some copy pasted code :)
        for layer in postprocessor_config:
            layer_params = postprocessor_config[layer]

            if layer == "resizing":
                postprocessor.add(tf.keras.layers.Resizing(**layer_params))
            elif layer == "resizepadcrop":
                postprocessor.add(layers.ResizePadCrop(**layer_params))
            # TODO add untiler
            # TODO add median filter
            else:
                raise NotImplementedError

        return postprocessor
