import tensorflow as tf
import os
import json


class Model:
    """Main Deep Neural network model template"""
    def __init__(self, dataset, network_name, input_shape, debug=True):
        """This create an instance for the Sermanet DNN model"""
        self.w, self.h, self.l = input_shape[0:3]
        self.debug = debug

        self.json = None
        self.model = None
        self.network_name = network_name

        self.dataset = dataset
        self.color_mode = 'rgb'

        self.train_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        self.validator_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    def train(self):
        """Training the model"""

        train_generator = self.train_idg.flow_from_directory(
            directory=self.dataset.train_path,
            classes=self.dataset.classNames,
            target_size=(self.w, self.h),
            batch_size=32,
            class_mode='categorical',
            color_mode=self.color_mode
        )

        validator_generator = self.validator_idg.flow_from_directory(
            directory=self.dataset.test_path,
            classes=self.dataset.classNames,
            target_size=(self.w, self.h),
            batch_size=32,
            class_mode='categorical',
            color_mode=self.color_mode
        )

        print(train_generator)

        # callbacks
        log_path = self.generateLogDir()
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_path,
            batch_size=32
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20
        )

        # train the model
        epochs = self.model.fit_generator(
            generator = train_generator,
            steps_per_epoch = 100,
            epochs = 30,
            validation_data = validator_generator,
            validation_steps = 50,
            callbacks = [
                tensorboard,
                early_stop
            ]
        )

        self.saveModel(log_path)

    def generateLogDir(self):
        run_counter = 0
        run_log_dir = os.path.join(self.dataset.program_path, "logs", self.network_name, "run{}".format(run_counter))

        while os.path.isdir(run_log_dir):
            run_counter += 1
            run_log_dir = os.path.join(self.dataset.program_path, "logs", self.network_name, "run{}".format(run_counter))

        return run_log_dir

    def saveModel(self, out_path):
        self.model.save(os.path.join(out_path, self.network_name + ".h5"))
        with open(os.path.join(out_path, self.network_name + ".json"), 'w') as f:
            f.write(
                json.dumps(
                    self.json,
                    indent = 4
                )
            )
        if not f.closed:
            f.close()