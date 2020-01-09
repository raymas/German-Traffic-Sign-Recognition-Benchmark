import argparse
import os

# dataset
from utils.dataset import GTSRB

# models
from models.dks import DKS
from models.sermanet import Sermanet

VALID_MODELS = {
    'DKS': DKS,
    'sermanet': Sermanet
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Select the model from the following : {}".format(", ".join(VALID_MODELS.keys())))
    parser.add_argument("--train", default=False, action='store_true', help="Start the model into train mode")
    parser.add_argument("--predict", help="Predict from input either an image ending with '.jpg', '.ppm', '.bmp', '.tiff', '.png' or a list of files path within in a '.csv' or '.txt'")
    parser.add_argument("--weights", help="Path of the h5 model file")
    arguments = parser.parse_args()

    gtsrb = GTSRB(os.path.abspath(os.path.dirname(__file__)))

    model = None

    if arguments.model not in VALID_MODELS:
        raise ValueError('Model is not part of valid model list')
    else:
        model = VALID_MODELS[arguments.model](gtsrb)

    if arguments.train:
            model.build()
            model.train()
    elif arguments.predict:
        if arguments.weights:
            model.loadModel(arguments.weights)
            model.predict(arguments.predict)
        else:
            raise ValueError("You must provide the --weights flag with predict")