import os
import sys
from urllib.request import urlretrieve
import tarfile
import pickle
import numpy as np
from scipy import ndimage


class DataDownload:
    def __init__(self):
        self.image_size = None
        self.pixel_depth = None
        pass

    def maybe_download(self, url, path, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        filepath = os.path.join(path, filename)
        if not os.path.exists(path):
            os.mkdir(path)
        if force or not os.path.exists(filepath):
            filename, _ = urlretrieve('/'.join([url, filename]), filepath)
        statinfo = os.stat(filepath)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filepath)
        else:
            raise Exception(
                'Failed to verify ' + filepath + '. Can you get to it with a browser?')
        return filepath

    def maybe_extract(self, filename, num_classes, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        extractpath = os.path.dirname(root)
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall(path=extractpath)
            tar.close()
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != num_classes:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (
                    num_classes, len(data_folders)))
        print(data_folders)
        return data_folders

    def load_letter(self, folder, min_num_images):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), self.image_size, self.image_size),
                             dtype=np.float32)

        num_images = 0
        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                image_data = ndimage.imread(image_file).astype(float) / self.pixel_depth
                if image_data.shape != (self.image_size, self.image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images, :, :] = image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        dataset = dataset[0:num_images, :, :]
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' %
                            (num_images, min_num_images))

        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))

        return dataset

    def maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = self.load_letter(folder, min_num_images_per_class)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

        return dataset_names

    def make_arrays(self, nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels

    def merge_datasets(self, pickle_files):
        assert len(pickle_files) == self.num_classes

        dataset, labels = np.empty([0, self.image_size, self.image_size], np.float32), np.empty(0, np.int32)

        for label, pickle_file in enumerate(pickle_files):
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                    dataset = np.concatenate([dataset, letter_set], axis=0)
                    labels  = np.concatenate([labels, np.tile(label, [len(letter_set)])], axis=0)

            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise
        # permute
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
        labels = labels[idx]

        return dataset, labels
