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
    arguments = parser.parse_args()

    gtsrb = GTSRB(os.path.abspath(os.path.dirname(__file__)))

    model = None
    if arguments.train:
        if arguments.model not in VALID_MODELS:
            raise ValueError('Model is not part of valid model list')
        else:
            model = VALID_MODELS[arguments.model](gtsrb)
            model.build()
            model.train()
