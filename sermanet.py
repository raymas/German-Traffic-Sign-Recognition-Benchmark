import tensorflow as tf


class Sermanet:
    """Main sermanet class"""

    def __init__(self, input_shape=(32, 32), debug=True):
        """This create an instance for the Sermanet DNN model"""
        self.input_shape = input_shape
        self.debug = debug

    def build(self):
        """ """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (1, 1), input_shape=(32, 32, 1), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(43)
        ])

        if self.debug:
            model.summary()

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(), 
            metrics=['accuracy']
        )

        return model

    def train(self):
        pass


if __name__ == '__main__':
    sermanet = Sermanet()
    sermanet.build()