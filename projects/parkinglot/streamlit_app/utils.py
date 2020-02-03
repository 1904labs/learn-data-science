import re
import datetime
import time

from collections import namedtuple
from typing import List, Callable, Union
from pathlib import Path

import cv2
import azure.storage.blob as blb
import dateparser

from azure.storage.blob._models import BlobProperties
from azure.cosmosdb.table.tableservice import TableService

ImageMeta = namedtuple('ImageMeta', 'fn room date')

class ImageLoader:
    
    def __init__(self, path: str='images'):
        self.image_path = Path(path)
        # self.images = sorted([str(fn) for fn in image_path.iterdir() if 'DS_Store' not in str(fn)])
        self._image_cache = {}
        # self._date_pat = re.compile('RM19(.*)|pike(.*)')  # Hard coded room names
        self.reset_image_list()
    
    # TODO: Break this image list management out into a separate class if it keeps getting bigger.
    def reset_image_list(self):
        """Read the current list of image available and parse date."""
        images = sorted((str(fn) for fn in self.image_path.iterdir() if 'DS_Store' not in str(fn)))
        images_meta = [ImageMeta(fn,
                                 self._parse_room(fn),
                                 self._parse_date(fn))
                        for fn in images]
        self.images_meta = images_meta


    def get_image_list(self, 
                        rooms: List[str]=None,
                        start_date: datetime.date=None,
                        end_date: datetime.date=None   
                        ):
        # images = sorted((str(fn) for fn in self.image_path.iterdir() if 'DS_Store' not in str(fn)))
        # room filter
        images = self.images_meta
        if rooms:
            images = (i_meta for i_meta in images if any(room==i_meta.room for room in rooms))
        # date range filter
        if start_date or end_date:
            if not start_date: start_date = datetime.datetime.min.date()
            if not end_date: end_date = datetime.datetime.max.date()
            images = (i_meta for i_meta in images if start_date <= i_meta.date <= end_date)
        return [i_meta.fn for i_meta in images]


    def _parse_date(self, fn: str) -> datetime.date:
        """Parse the datetime out of a filename."""
        # Try regex here and failed.
        # str_date = self._date_pat.findall(fn)[0][1].split('.')[0].replace('_', ' ').strip()
        if 'RM19' in fn:
            str_date = fn.partition('RM19')[-1].split('.')[0].replace('_', ' ').strip()
        else:
            str_date= fn.partition('pike')[-1].split('.')[0].replace('_', ' ').strip()
        return dateparser.parse(str_date).date()

    def _parse_room(self, fn: str) -> str:
        """Determine room from image filename."""
        if 'RM19' in fn:
            return 'RM19'
        elif 'pike' in fn:
            return 'pike'
        else:
            raise ValueError(f'Unknown room {fn}')


    def get_image(self, fn: Union[str, Path]):
        """Return image as Numpy array, checking cache for previously loaded images."""
        fn = str(fn)
        img = self._image_cache.get(fn)
        if img is None:
            img = self.load_image(fn)
            self._image_cache[fn] = img
        return img
    
    
    def load_image(self, fn: str):
        """Load image file using OpenCV."""
        img = cv2.imread(str(fn), 1)
        img = img[:, :, ::-1]
        img = cv2.flip(img, 0) # Move out of this loader
        img = cv2.flip(img, 1) # Move out of this loader
        return img
    



class BlobDownloader:
    """Object to interact with blob storage to download images. 
    
    There is intentionally no caching. Calling code should decide what/when
    to cache lists. Calls to this object will always read from blob.
    """
    
    def __init__(self,
                 account_url: str,
                 container_name: str,
                 sas_token: str
                ):
        self._account_url = account_url
        self._container_name = container_name
        self._sas_token = sas_token
        self._container_client = self._create_container_client()
        
        
    def _create_container_client(self):
        """Azure container client object."""
        return blb.ContainerClient(self._account_url, 
                                   self._container_name, 
                                   self._sas_token)
    
    
    def image_list(self):
        """Return list of BlobProperties for all blobs in container."""
        return list(self._container_client.list_blobs())


    def filtered_image_list(self, filters:List[Callable[[BlobProperties], bool]]=None):
        """Return list of blobs filtered by functions in `filters`.
        
        Each function in `filters` should accept a BlobProperties object and return True
        if that blob's name should be included in the list.
        """
        if filters is None:
            return self.get_blob_names_from_props(self.image_list())
        else:
            images = self.image_list()
            for cll in filters:
                images = [im for im in images if cll(im)]
            return self.get_blob_names_from_props(images)
    
    
    def get_daytime_image_list(self):
        """Convenience function for getting current list of daytime blobs."""
        return self.filtered_image_list(filters=[self.image_is_daytime])
    
    
    def download_image(self, blob_name: str, path: Union[str, Path]):
        """Download a blob (image) to `path`."""
        b = blb.BlobClient(account_url=self._account_url, 
                       container_name=self._container_name, 
                       blob_name=blob_name, 
                       credentials=self._sas_token)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / blob_name, 'wb') as f:
            f.write(b.download_blob().readall())
            
    
    def download_images(self, blobs: List[str], path: Union[str, Path]):
        """Download a list of blobs to `path`."""
        for blob_name in blobs:
            try:
                self.download_image(blob_name, path)
            except Exception as e:
                print(f'Downloading {blob_name} failed!')
                print(e)        
        
        
    @staticmethod
    def image_is_daytime(prop: BlobProperties) -> bool:
        """Given an image BlobProperties, return true if daytime image."""
        return 8 < prop['creation_time'].hour - 5 and prop['creation_time'].hour - 5 < 17
    
    
    @staticmethod
    def get_blob_names_from_props(props: List[BlobProperties]) -> List[str]:
        """Extract and return the blob names."""
        return [prop['name'] for prop in props]


class TableSaver:

    def __init__(self,
                 account_name: str,
                 sas_token: str,
                 table_name: str
                ):
        self.account_name = account_name
        self.table_name = table_name
        self.sas_token = sas_token
        self.table_service = TableService(account_name=account_name, sas_token=sas_token)


    def save_count(self, fn: str, count: int):
        """Save a manual count of cars for image `fn` to Azure table."""
        new_count = {'PartitionKey': Path(fn).name, 
                     'RowKey': str(time.time()),
                     'count': count}
        self.table_service.insert_entity(self.table_name, new_count)