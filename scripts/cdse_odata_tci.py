import requests
import json
from tqdm import tqdm
import os.path
from getpass import getpass
import tempfile
import zipfile
import logging
import rasterio as rio
from rasterio.io import MemoryFile
import io

from fastcore.script import *

class FileDownloader:
    # initialize logger
    logger = logging.getLogger(__name__)

    def __init__(self, username=None, password=None, creds_file=None):

        if username is not None and password is not None:
            data = {'grant_type': 'password', 'username': username, 'password': password, 'client_id': 'cdse-public'}
        if creds_file is not None:
            self.logger.debug(f'Reading creds from external file {creds_file}.')
            with open(creds_file, 'r') as fp:
                data = json.load(fp)
        else:
            self.logger.debug('No init arguments, query password interactively.')
            username = input('Username: ')
            password = getpass()
            data = {'grant_type': 'password', 'username': username, 'password': password, 'client_id': 'cdse-public'}

        self.token = self.get_keycloak_token(data)
        self.response = None

    def get_keycloak_token(self, data):
        url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
        except Exception as e:
            raise Exception(
                f'Keycloak token creation failed. Response from the server was: {response.json()}'
            )
        return response.json()['access_token']

    def download_latest_response(self, target_path='/tmp'):
        for idx, val in enumerate(self.response['value']):
            file_id = val['Id']
            name = val['Name']
            self.logger.info(f'Downloading file #{idx + 1}...')
            self.logger.info(json.dumps(val, indent=4))
            self.download_file(file_id, target_path, name)

    def download_file(self, file_id, target_path, name=None):
        headers = {'Authorization': f'Bearer {self.token}'}
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({file_id})/Nodes({name})/Nodes(GRANULE)/Nodes"
        response = requests.get(url, headers=headers, stream=True)
        t = json.loads(response.text)
        parts = name[:-5].split('_')
        identifier = '_'.join([parts[5], parts[2]])
        next_url = t['result'][0]['Nodes']['uri'] + f'(IMG_DATA)/Nodes(R10m)/Nodes({identifier}_TCI_10m.jp2)/$value'
        response = requests.get(f'{next_url}', headers=headers, stream=True)
        im = io.BytesIO(response.content)
        with MemoryFile(im) as memfile:
            with memfile.open() as src:
                data = src.read()
                prof = src.profile
                prof.update({'driver':'GTiff',
                             'QUALITY':'100',
                             'REVERSIBLE':'YES'})
                with rio.open(f'{target_path}/{name[:-5]}_TCI_10m.tif', 'w', **prof) as dest:
                    dest.write(data)
        return 

    def query_product_by_name(self, name):
        url = f'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq \'{name}\''
        response = requests.get(url)  # doesn't require token authentication
        self.response = json.loads(response.text)
        return json.loads(response.text)


@call_parse
def main(product_name:str, # Product name to download
         username:str=None, # Username
         password:str=None, # Password
         creds_file:str=None, # Path to credential files
         target_path:str='tmp'): # Target path, default 'tmp'
    "Download `product_name` to `target_path`"
    dl = FileDownloader(username=username, password=password, creds_file=creds_file)
    dl.query_product_by_name(name=product_name)
    dl.download_latest_response(target_path=target_path)