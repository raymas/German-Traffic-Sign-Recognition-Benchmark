import tensorflow as tf
from models.dnn_model import Model


class DKS(Model):
    """Deep Knowledge Seville's neural network class"""

    def __init__(self, dataset, network_name='DKS', input_shape=(48, 48, 3), debug=True):
        super().__init__(dataset, network_name=network_name, input_shape=input_shape, debug=debug)

    def build(self):
        """ """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.w, self.h, self.l)),
            tf.keras.layers.MaxPool2D((2, 2), padding='valid'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), padding='valid'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), padding='valid'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(43, activation='softmax')
        ])

        if self.debug:
            model.summary()

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.RMSprop(), 
            metrics=['accuracy']
        )

        self.json = model.to_json()

        self.model = model
