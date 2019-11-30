import tensorflow as tf
import numpy as np

from models.dnn_model import Model


class Sermanet(Model):
    """Main sermanet class"""

    def __init__(self, dataset, network_name="Sermanet", input_shape=(32, 32, 1), debug=True):
        super().__init__(dataset, network_name=network_name, input_shape=input_shape, debug=debug)

        # on rgb for training
        self.color_mode = 'grayscale'

    def build(self):
        """THIS MODEL IS NOT ACCURATE FOR NOW!"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (1, 1), input_shape=(self.w, self.h, self.l), activation='relu'),
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

        print("WARNING : NETWORK NOT ACCURATE !!!")

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(), 
            metrics=['accuracy']
        )

        self.model = model
