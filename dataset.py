import os
import sys
import urllib.request
import zipfile

# from PIL import Image

class GTSRB:

    def __init__(self):
        """Main class for training set.
        - Able to download training and testing sets
        - Retrieves anotation from attached database csv"""
        base_dir = os.path.join(os.path.dirname(__file__), "GTSRB")

        train_link, train_filename, train_folder = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Training_Images.zip", "Final_Training"
        test_link, test_filename, test_folder = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Test_Images.zip", "Final_Test"
        gtd_link, gtd_filename, gtd_extracted = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Test_GT.zip", "GT-final_test.csv"

        bFolderExists = os.path.isdir(base_dir)
        
        self.train_dataset = None
        self.train_path = os.path.abspath(os.path.join(base_dir, train_folder, "Images"))

        self.test_dataset = None
        self.test_path = os.path.abspath(os.path.join(base_dir, test_folder, "Images"))

        self.classNames = []

        if not bFolderExists:
            os.mkdir(base_dir)

        self.checkForDataset(base_dir, train_folder, train_link, train_filename)
        self.checkForDataset(base_dir, test_folder, test_link, test_filename)

    def getAnnotations(self, folder):
        """Extract annotations from csv files"""
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "GTSRB", folder)
        for dirpath, dirnames, filenames in os.walk(path, topdown=False):
            for filename in filenames:
                if filename.endswith('.csv'):
                    pass

    def checkForDataset(self, path, folder, link, filename):
        """Check if dataset is present. If not download and anotates files"""
        bIsDownloaded = os.path.isdir(os.path.join(path, folder))
        if not bIsDownloaded:
            GTSRB.downloadFile(link, filename)
            print("Unzipping {}".format(filename))

            with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), filename), 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(__file__))

        self.getAnnotations(os.path.join(os.path.abspath(path), folder))

    def checkForGroundThruth(self, path, folder, link, filename):
        bIsDownloaded = os.path.isfile(os.path.join(path, folder))
        if not bIsDownloaded:
            GTSRB.downloadFile(link, filename)
            print("Unzipping {}".format(filename))

            with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), filename), 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(__file__))

        self.getAnnotations(os.path.join(os.path.abspath(path), folder))        

    def readClassNames(self):
        file_path = os.path.abspath(os.path.dirname(__file__), "classnames.txt")

        with open(file_path, 'r') as f:
            self.classNames = f.readlines().split('\n')
        if not f.closed:
            f.close()

    @staticmethod
    def downloadFile(link, file_name):
        u = urllib.request.urlopen(link + file_name)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.get("Content-Length"))

        print("Downloading {}".format(file_name))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            sys.stdout.write(status)
            sys.stdout.flush()

        f.close()


    class Sign:
        def __init__(self, id, coordinates, roi):
            self.id = id
            self.coordinates = coordinates
            self.roi = roi        

if __name__ == "__main__":
    GTSRB()