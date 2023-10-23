import tensorflow as tf

class AugmenterBuilder:
    @staticmethod
    def augmenter_from_config(
        augmenter_config: dict = {}, seed: int = None
    ) -> tf.keras.Sequential:
        augmenter = tf.keras.Sequential()

        if len(augmenter_config) == 0:
            print("Skipping augmenter creation. Empty config")
            return

        # NOTE not really proud of this part of code probably preprocessors and augmenters
        # are things will probably need a refactor in the future.
        # Things I am considering: strategy pattern with predefined augmenters
        # and ability to add custom augmenters
        # But I don't want to overcomplicate things right now
        for layer in augmenter_config:

            layer_params = augmenter_config[layer]

            if layer == "flip":
                augmenter.add(tf.keras.layers.RandomFlip(**layer_params, seed=seed))
            elif layer == "rotation":
                augmenter.add(tf.keras.layers.RandomRotation(**layer_params, seed=seed))
            elif layer == "translation":
                augmenter.add(
                    tf.keras.layers.RandomTranslation(**layer_params, seed=seed)
                )
            else:
                raise NotImplementedError

        return augmenter
