import os
import sys
import urllib.request
import zipfile
import pandas

# from PIL import Image

class GTSRB:

    def __init__(self, program_path):
        """Main class for training set.
        - Able to download training and testing sets
        - Retrieves anotation from attached database csv"""
        self.program_path = program_path
        self.current_path = os.path.abspath(os.path.dirname(__file__))
        self.dataset_dir = os.path.join(program_path, "GTSRB")

        train_link, train_filename, train_folder = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Training_Images.zip", "Final_Training"
        test_link, test_filename, test_folder = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Test_Images.zip", "Final_Test"
        gtd_link, gtd_filename, gtd_extracted = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/", "GTSRB_Final_Test_GT.zip", "GT-final_test.csv"

        bFolderExists = os.path.isdir(self.dataset_dir)
        
        self.train_dataset = None
        self.train_path = os.path.abspath(os.path.join(self.dataset_dir, train_folder, "Images"))

        self.test_dataset = None
        self.test_path = os.path.abspath(os.path.join(self.dataset_dir, test_folder, "Images"))

        self.classNames = None

        if not bFolderExists:
            os.mkdir(self.dataset_dir)

        # check and download datatset + extended class ground truth.
        self.checkForDataset(self.dataset_dir, train_folder, train_link, train_filename)
        self.checkForDataset(self.dataset_dir, test_folder, test_link, test_filename)

        self.checkForGroundThruth(self.dataset_dir, test_folder, gtd_link, gtd_filename, gtd_extracted)
        self.classTestDataset(gtd_extracted)
        self.readClassNames()

    def getAnnotations(self, folder, target=None):
        """Extract annotations from csv files"""
        path = os.path.join(self.program_path, "GTSRB", folder)
        for dirpath, dirnames, filenames in os.walk(path, topdown=False):
            for filename in filenames:
                if filename.endswith('.csv'):
                    if target is not None:
                        if filename == target:
                            df = pandas.read_csv(os.path.join(dirpath, filename), sep=';', quotechar='|')
                            return df
                    else:
                        # pass for now
                        pass

    def checkForDataset(self, dataset_dir, folder, link, filename):
        """Check if dataset is present. If not download and anotates files"""
        bIsDownloaded = os.path.isdir(os.path.join(dataset_dir, folder))
        if not bIsDownloaded:
            GTSRB.downloadFile(link, filename)
            downloaded_path = os.path.join(self.program_path, filename)
            
            GTSRB.unzip(downloaded_path, self.program_path)
            os.remove(downloaded_path)

        self.getAnnotations(os.path.join(os.path.abspath(dataset_dir), folder))

    def checkForGroundThruth(self, dataset_dir, folder, link, filename, filename_extracted):
        bIsDownloaded = os.path.isfile(os.path.join(dataset_dir, folder, "Images", filename_extracted))
        if not bIsDownloaded:
            GTSRB.downloadFile(link, filename)
            downloaded_path = os.path.join(self.program_path, filename)

            GTSRB.unzip(downloaded_path, self.program_path)
            os.remove(downloaded_path)

            os.rename(os.path.join(self.program_path, filename_extracted), os.path.join(self.test_path, filename_extracted))

    def classTestDataset(self, csv_filename):
        """Create the directory structure for test validator"""
        df = self.getAnnotations(self.test_path, csv_filename)
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

    @staticmethod
    def unzip(file_path, destination_path):
        """Unzip a file"""
        print("Unzipping {}".format(file_path))
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_path)

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