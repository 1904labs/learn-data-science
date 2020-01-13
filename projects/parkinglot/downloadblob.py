import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

AZURE_CONNECTION_STRING = '*'
local_path = 'parkinglotimages/'
container_url = 'blob_url'


try:
    blob_client = ContainerClient.from_container_url(container_url) #BlobClient.from_connection_string(conn_str=AZURE_CONNECTION_STRING, container_name='images', blob_name='*')

    blobList = blob_client.list_blobs(None, None)
    #print(len(blobList))
    # Quick start code goes here
    for blob in blobList:
        print(blob.name)
        # Download the blob to a local file
        # Add 'DOWNLOAD' before the .txt extension so you can see both files in Documents
        download_file_path = os.path.join(local_path, blob.name )
        print("\nDownloading blob to \n\t" + download_file_path)

        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob(blob).readall())
        
except Exception as ex:
    print('Exception:')
    print(ex)

