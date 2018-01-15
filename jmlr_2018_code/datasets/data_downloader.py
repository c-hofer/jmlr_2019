import tempfile
import zipfile
import requests
import os
import time

from pathlib import Path


_data_set_name_2_provider_google_drive_id = \
    {
        'animal': '0BxHF82gaPzgSSWIxNmJBRFJzcmM',
        'mpeg7': '0BxHF82gaPzgSU3lPWDNEVHhNR3M',
        'reddit_5K': '0BxHF82gaPzgSZDdFWDU3S29hdm8',
        'reddit_12K': '0BxHF82gaPzgSd0d4WDNYVnN4dEU',
    }


_data_set_name_2_provider_name = \
    {
        'animal': 'npht_animal_32dirs.h5',
        'mpeg7': 'npht_mpeg7_32dirs.h5',
        'reddit_5K': 'reddit_5K.h5',
        'reddit_12K': 'reddit_12K.h5'
    }


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_iter = response.iter_content(CHUNK_SIZE)
        with open(destination, "wb") as f:

            for i, chunk in enumerate(content_iter):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    print(i, end='\r')

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def ensure_path_existence(the_path):
    parent_dir = os.path.dirname(the_path)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
