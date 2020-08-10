import os
import sys
import urllib.request
import zipfile
import pandas
import tensorflow as tf

# from PIL import Image

class GTSRB:

    def __init__(self, program_path):
        """Main class for training set.
        - Able to download training and testing sets
        - Retrieves anotation from attached database csv"""
        self.program_path = program_path

        train_file = tf.keras.utils.get_file(
            origin="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
            fname="GTSRB_Final_Training_Images.zip",
            extract=True
        )

        tf.keras.utils.get_file(
            origin="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
            fname="GTSRB_Final_Test_Images.zip",
            extract=True
        )

        gtd_file = tf.keras.utils.get_file(
            origin="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
            fname="GTSRB_Final_Test_GT.zip",
            extract=True
        )

        self.dataset_dir = os.path.join(os.path.split(train_file)[0], 'GTSRB')
        
        self.train_dataset = None
        self.train_path = os.path.abspath(os.path.join(self.dataset_dir, 'Final_Training', 'Images'))

        self.test_dataset = None
        self.test_path = os.path.abspath(os.path.join(self.dataset_dir, 'Final_Test', 'Images'))

        self.gtd_path = os.path.split(gtd_file)[0]

        self.classNames = None

        self.classTestDataset('GT-final_test.csv')
        self.readClassNames()

    def getAnnotations(self, folder, target=None):
        """Extract annotations from csv files"""
        for dirpath, _, filenames in os.walk(folder, topdown=False):
            for filename in filenames:
                if filename.endswith('.csv'):
                    if target is not None:
                        if filename == target:
                            df = pandas.read_csv(os.path.join(dirpath, filename), sep=';', quotechar='|')
                            return df
                    else:
                        # pass for now
                        pass

    def classTestDataset(self, csv_filename):
        """Create the directory structure for test validator"""
        df = self.getAnnotations(self.gtd_path, csv_filename)
        nb_of_class = max(df['ClassId']) + 1

        for class_id in range(nb_of_class):
            dir_path = os.path.join(self.test_path, str(class_id).zfill(5))
            bDirPresent = os.path.isdir(dir_path)

            if not bDirPresent:
                os.mkdir(dir_path)
                associated_images = df[(df['ClassId'] == class_id)]
                GTSRB.moveImages(associated_images['Filename'], self.test_path, dir_path)
                

    def readClassNames(self):
        file_path = os.path.abspath(os.path.join(self.program_path, "classnames.txt"))

        with open(file_path, 'r') as f:
            classNames = f.readlines()
        if not f.closed:
            f.close()

        self.classNames = [c.replace('\n', '') for c in classNames if c]

    def getStats(self):
        pass

    @staticmethod
    def moveImages(images_list, base_dir, final_dir):
        """Moving images list from base directory to final directory. Name remains unchanged"""
        for image in images_list:
            os.rename(os.path.join(base_dir, image), os.path.join(final_dir, image))

    class Sign:
        def __init__(self, id, coordinates, roi):
            self.id = id
            self.coordinates = coordinates
            self.roi = roi        

if __name__ == "__main__":
    GTSRB(os.path.join('..', os.path.dirname(__file__)))