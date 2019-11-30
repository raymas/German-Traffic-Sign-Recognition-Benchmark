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
        """Building the main sermanet model"""

        input_layer = tf.keras.layers.Input(shape=(self.w, self.h, self.l))

        # cnn 1 -> flatten
        cnn_1 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
        max_pool_1 = tf.keras.layers.MaxPool2D((2, 2), padding='valid')(cnn_1)
        dropout_1 = tf.keras.layers.Dropout(0.9)(max_pool_1)

        # cnn 2 -> flatten
        cnn_2 = tf.keras.layers.Conv2D(64, (1, 1) ,activation='relu')(dropout_1)
        max_pool_2 = tf.keras.layers.MaxPool2D((2, 2), padding='valid')(cnn_2)
        dropout_2 = tf.keras.layers.Dropout(0.8)(max_pool_2)

        # cnn 3 -> flatten
        cnn_3 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu')(dropout_2)
        max_pool_3 = tf.keras.layers.MaxPool2D((2, 2), padding='valid')(cnn_3)
        dropout_3 = tf.keras.layers.Dropout(0.7)(max_pool_3)

        # flatten first layer
        max_pool_1_2 = tf.keras.layers.MaxPool2D((4, 4), padding='valid')(dropout_1)
        flatten_1 = tf.keras.layers.Flatten()(max_pool_1_2)

        # flatten second layer
        max_pool_2_2 = tf.keras.layers.MaxPool2D((2, 2), padding='valid')(dropout_2)
        flatten_2 = tf.keras.layers.Flatten()(max_pool_2_2)

        # flatten third layer
        flatten_3 = tf.keras.layers.Flatten()(dropout_3)

        # Merge layers
        merged = tf.keras.layers.Concatenate(axis=1)([flatten_1, flatten_2, flatten_3])

        # Dense reduction
        dense_1 = tf.keras.layers.Dense(1024)(merged)
        dropout_4 = tf.keras.layers.Dropout(0.5)(dense_1)
        
        output_layer = tf.keras.layers.Dense(43)(dropout_4)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        if self.debug:
            model.summary()

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(), 
            metrics=['accuracy']
        )

        self.model = model
