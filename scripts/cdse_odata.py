import requests
import json
from tqdm import tqdm
import os.path
from getpass import getpass
import tempfile
import zipfile
import logging

from fastcore.script import *
from fastcore.basics import *

class FileDownloader:
    # initialize logger
    logger = logging.getLogger(__name__)

    def __init__(self, username=None, password=None, creds_file=None):
        store_attr()
        self.token_data = self.generate_token_data(self.username, self.password, self.creds_file)
        self.session = requests.Session()
        self.token, self.rtoken = self.get_keycloak_token(self.token_data)
        self.response = None

    def generate_token_data(self, username=None, password=None, creds_file=None):
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
        return data

    def get_keycloak_token(self, data):
        url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        try:
            response = self.session.post(url, data=data)
            response.raise_for_status()
        except Exception as e:
            raise Exception(
                f'Keycloak token creation failed. Response from the server was: {response.json()}'
            )
        access_token = response.json()['access_token']
        rtoken = response.json()['refresh_token']
        return access_token, rtoken

    def refresh_token(self):
        url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        data = {'grant_type': 'refresh_token', 'refresh_token': self.rtoken, 'client_id': 'cdse-public'}
        try:
            response = self.session.post(url, data)
            response.raise_for_status()
            self.token = response.json()['access_token']
            self.rtoken = response.json()['refresh_token']
        except Exception as e:
            self.logger.info(f'Refreshing token failed. Response from the server was: {response.json}')
            self.logger.info('Creating new keycloak token')
            access_token, rtoken = self.get_keycloak_token(self.token_data)
            self.token = access_token
            self.rtoken = rtoken
        return

    def download_latest_response(self, target_path='/tmp'):
        for idx, val in enumerate(self.response['value']):
            file_id = val['Id']
            name = val['Name']
            self.logger.info(f'Downloading file #{idx + 1}...')
            self.logger.info(json.dumps(val, indent=4))
            self.download_file(file_id, target_path, name)

    def download_file(self, file_id, target_path, name=None):
        headers = {'Authorization': f'Bearer {self.token}'}
        url = f'https://zipper.dataspace.copernicus.eu/odata/v1/Products({file_id})/$value'
        response = self.session.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.join(target_path, name), ascii=True)
        num_retries = 0
        while num_retries < 5:
            try:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as temp_zip:
                    with open(temp_zip.name, 'wb') as f:
                        for chunk in response.iter_content(block_size):
                            t.update(len(chunk))
                            f.write(chunk)
                    self.logger.info('Extracting...')
                    with zipfile.ZipFile(temp_zip.name, 'r') as zip_ref:
                        zip_ref.extractall(target_path)
            except:
                num_retries += 1
                self.logger.info(f'{target_path} failed, retry {num_retries}/5')
            else:
                break
        return os.path.join(target_path, name)

    def query_product_by_name(self, name):
        url = f'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq \'{name}\''
        response = self.session.get(url)  # doesn't require token authentication
        self.response = json.loads(response.text)
        return json.loads(response.text)

@call_parse
def cdse_odata(product_name:str, # Product name to download
               username:str=None, # Username
               password:str=None, # Password
               creds_file:str=None, # Path to credential files
               target_path:str='tmp'): # Target path, default 'tmp'
    "Download `product_name` to `target_path`"
    dl = FileDownloader(username=username, password=password, creds_file=creds_file)
    dl.query_product_by_name(name=product_name)
    dl.download_latest_response(target_path=target_path)