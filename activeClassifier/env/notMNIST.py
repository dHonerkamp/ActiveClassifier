import os
import numpy as np

from activeClassifier.env.dataDownload import DataDownload

class notMNIST(DataDownload):
    def __init__(self, datadir, test_only=False):
        super().__init__()
        # to be implemented by children
        self.name = 'notMNIST'
        self.url = 'http://yaroslavvb.com/upload/notMNIST/'
        self.num_classes = 10
        self.image_size = 28
        self.pixel_depth = 255.0 # Number of levels per pixel.

        self.train_size = 200000
        self.valid_size = 10000

        if not test_only:
            train_filename = self.maybe_download(self.url, os.path.join(datadir, self.name), 'notMNIST_large.tar.gz', 247336696)
            train_folders = self.maybe_extract(train_filename, self.num_classes)
            train_datasets = self.maybe_pickle(train_folders, min_num_images_per_class=45000)
            dataset, labels = self.merge_datasets(train_datasets)
            train_dataset, train_labels = dataset[:self.train_size], labels[:self.train_size]
            valid_dataset, valid_labels = dataset[self.train_size:self.valid_size], labels[self.train_size:self.valid_size]
        else:
            train_dataset, train_labels = None, None
            valid_dataset, valid_labels = None, None
        test_filename = self.maybe_download(self.url, os.path.join(datadir, self.name), 'notMNIST_small.tar.gz', 8458043)
        test_folders  = self.maybe_extract(test_filename, self.num_classes)
        test_datasets  = self.maybe_pickle(test_folders, min_num_images_per_class=1800)

        test_dataset, test_labels = self.merge_datasets(test_datasets)
        test_dataset = np.expand_dims(test_dataset, -1)

        self.train = (train_dataset, train_labels)
        self.valid = (valid_dataset, valid_labels)
        self.test = (test_dataset, test_labels)
