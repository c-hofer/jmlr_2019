"""
This module contains a nice wrapper for the persistence barcode data sets used in

@inproceedings{Hofer17c,
  author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
  title     = {Deep Learning with Topological Signatures},
  booktitle = {NIPS},
  year      = 2017}

The code how the barcodes were generated from the origional data sets can be found on

    https://github.com/c-hofer/nips2017.

"""
import os
import os.path as pth

from .data_downloader import download_file_from_google_drive
from .provider import Provider

from sklearn.preprocessing import LabelEncoder


class DataSetException(Exception):
    pass


class DataSetBase:
    google_drive_provider_id = None
    provider_file_name = None

    def __init__(self, root_dir: str, download=True, sample_transforms=[]):
        self.root_dir = pth.normpath(root_dir)
        self.sample_transforms = sample_transforms
        self._provider = None
        self.integer_labels = True

        provider_exists = pth.isfile(self._provider_file_path)

        if not provider_exists:
            print("Did not find data in {}!".format(self.root_dir))

            if download:
                print("Downloading ... ")
                if not pth.isdir(self.root_dir):
                    os.mkdir(self.root_dir)

                download_file_from_google_drive(self.google_drive_provider_id,
                                                self._provider_file_path)

        provider_exists = pth.isfile(self._provider_file_path)
        if provider_exists:
            print('Found data!')
            self._provider = Provider()
            self._provider.read_from_h5(self._provider_file_path)

        else:
            raise DataSetException("Cannot find data in {}.".format(self.root_dir))

        self.str_2_int_label = {str_label: int_label for int_label, str_label in enumerate(self._provider.labels)}

    @property
    def _provider_file_path(self):
        return pth.join(self.root_dir, self.provider_file_name)

    def __getitem__(self, item):
        x, y = self._provider[item]

        for t in self.sample_transforms:
            x = t(x)

        if self.integer_labels:
            y = self.str_2_int_label[y]

        return x, y

    def __len__(self):
        return len(self._provider)

    @property
    def labels(self):
        return [self[i][1] for i in range(len(self))]


class Animal(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSSWIxNmJBRFJzcmM'
    provider_file_name = 'npht_animal_32dirs.h5'


class Mpeg7(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSU3lPWDNEVHhNR3M'
    provider_file_name = 'npht_mpeg7_32dirs.h5'


class Reddit_5K(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSZDdFWDU3S29hdm8'
    provider_file_name = 'reddit_5K.h5'


class Reddit_12K(DataSetBase):
    google_drive_provider_id = '0BxHF82gaPzgSd0d4WDNYVnN4dEU'
    provider_file_name = 'reddit_12K.h5'









