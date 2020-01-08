import time
import azure.storage.blob as blb

conn_str = 'azure connection string'
container_name = 'images'

SLEEP = 5

from io import BytesIO
from time import sleep
from picamera import PiCamera

import sys
print(sys.version_info)

while True:
    # Create the in-memory stream
    stream = BytesIO()
    camera = PiCamera()
#    camera.start_preview()
#    sleep(2)
    camera.capture(stream, format='jpeg')
    time_now = time.ctime().replace(' ', '_')

    # "Rewind" the stream to the beginning so we can read its content
    stream.seek(0)

    # Create blob client
    blob_name = f'parkinglot_{time_now}.jpeg'
    blob_client = blb.BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=blob_name)

    # Save image to blob
    blob_client.upload_blob(stream)

    print(f'file {blob_name} uploaded.')

    # Sleep
    time.sleep(SLEEP * 60)