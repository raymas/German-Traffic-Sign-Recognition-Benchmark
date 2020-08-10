import tensorflow as tf
import os
import json
from PIL import Image
import numpy as np


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
            directory=os.path.abspath(self.dataset.train_path),
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

    def loadModel(self, h5_path):
        self.model = tf.keras.models.load_model(
            h5_path,
            custom_objects=None,
            compile=True
        )

    def prepareSingleImage(self, image_path):
        """Convert an image path to a correct numpy array for prediction"""
        image = Image.open(image_path)
        image = image.resize((self.w, self.h), Image.ANTIALIAS)
        image = np.array(image)
        image.shape = (1, self.w, self.h, self.l)
        return image

    def predict(self, input_data):

        if isinstance(input_data, str):
            file_list = None
            if input_data.endswith(('.jpg', '.ppm', '.bmp', '.tiff', '.png')):
                file_list = [input_data]
            elif input_data.endswith(('.csv', '.txt')):
                with open(input_data, 'r') as f:
                    file_list = [x for x in f.readlines().split('\n') if x]
                    print(file_list)
                if not file_list:
                    raise ValueError("Error while reading file list")

            for file in file_list:
                image = self.prepareSingleImage(file)
                prediction = self.model.predict(image)
                potential_classes = np.where(np.any(prediction > 0.0, axis=0))
                confidences_level = prediction[0][potential_classes] * 100.
                sorted_index = np.argsort(np.sort(confidences_level)[::-1])
                
                separator = "{}".format("".join(["-" for _ in range(21 + len(file))]))

                print(separator)
                print("--- Prediction : {} ---".format(file))
                if self.dataset.classNames:
                    for c in sorted_index:
                        print("[class: {:<50} | confidence: {}%]".format(self.dataset.classNames[potential_classes[0][c]], confidences_level[c]))
                else:
                    for c in sorted_index:
                        print("[class: {:<2} | confidence: {}%]".format(potential_classes[0][c], confidences_level[c]))
                print(separator)

            