import tensorflow as tf
import os
import json

from dataset import GTSRB


class DKS:
    """Deep Knowledge Seville's neural network class"""

    def __init__(self, dataset, input_shape=(48, 48, 3), debug=True):
        """This create an instance for the Sermanet DNN model"""
        self.w, self.h, self.l = input_shape[0:3]
        self.debug = debug

        self.json = None
        self.model = None

        self.dataset = dataset

        self.train_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        self.validator_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

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

    def train(self):
        """Training the model"""

        train_generator = self.train_idg.flow_from_directory(
            directory=self.dataset.train_path,
            classes=self.dataset.classNames,
            target_size=(48, 48),
            batch_size=32,
            class_mode='categorical'
        )

        validator_generator = self.validator_idg.flow_from_directory(
            directory=self.dataset.test_path,
            classes=self.dataset.classNames,
            target_size=(48, 48),
            batch_size=32,
            class_mode='categorical'
        )

        # callbacks
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='./logs/DKS',
            batch_size=32
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            # TODO: change the monitor
            monitor='val_loss',
            patience=20
        )

        # train the model
        epochs = self.model.fit_generator(
            generator = train_generator,
            steps_per_epoch = 100,
            epochs = 30,
            # validation_data = validator_generator,
            # validation_steps = 50
            callbacks = [
                tensorboard,
                early_stop
            ]
        )

        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "DKS"))

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        self.model.save(os.path.join(out_path, "DKS.h5"))
        with open(os.path.join(out_path, "DSK.json"), 'w') as f:
            f.write(
                json.dumps(
                    self.json,
                    indent = 4
                )
            )
        if not f.closed:
            f.close()



if __name__ == '__main__':
    dks = DKS(dataset=GTSRB())
    dks.build()
    dks.train()