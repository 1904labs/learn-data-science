import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import sys
import time
import glob
import os

import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

import pandas as pd
from PIL import ImageOps
from keras.preprocessing import image

from keras import backend as K
import cv2


def keras_setup():

    from keras.preprocessing import image

    ### Limit GPU RAM initially grabbed by TensorFlow
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session  

    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
    # tf_config.gpu_options.per_process_gpu_memory_fraction=0.333
    sess = tf.Session(config=tf_config)  
    set_session(sess)  # set this TensorFlow session as the default session for Keras.
    return sess

sess = keras_setup()



# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)



# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

def get_model():
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model 

model = get_model()
K.set_session(sess)


def load_image_cv2(fn):
    """Load image file using OpenCV"""
    img = cv2.imread(str(fn), 1)
    img = img[:, :, ::-1]
    img = cv2.flip(img, 0)
    return img


AZURE_CONNECTION_STRING = '*'
local_path = 'images/'
container_url = '<fill it>'

def downloadAll():
    try:
        blob_client = ContainerClient.from_container_url(container_url) #BlobClient.from_connection_string(conn_str=AZURE_CONNECTION_STRING, container_name='images', blob_name='*')

        blobList = blob_client.list_blobs(None, None)
        #print(len(blobList))
        # Quick start code goes here
        for blob in blobList:
            #print(blob.name)
            # Download the blob to a local file
            # Add 'DOWNLOAD' before the .txt extension so you can see both files in Documents
            download_file_path = os.path.join(local_path, blob.name )
            #print("\nDownloading blob to \n\t" + download_file_path)
            #if os.path.exists(download_file_path) != True :
            print("\nDownloading blob to \n\t" + download_file_path)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob(blob).readall())
                
    except Exception as ex:
        print('Exception:')
        print(ex)


def downloadLatest():
    try:
        blob_client = ContainerClient.from_container_url(container_url) #BlobClient.from_connection_string(conn_str=AZURE_CONNECTION_STRING, container_name='images', blob_name='*')

        blobList = blob_client.list_blobs(None, None)
        #print(len(blobList))
        # Quick start code goes here
        for blob in blobList:
            #print(blob.name)
            # Download the blob to a local file
            # Add 'DOWNLOAD' before the .txt extension so you can see both files in Documents
            download_file_path = os.path.join(local_path, blob.name )
            #print("\nDownloading blob to \n\t" + download_file_path)
            if os.path.exists(download_file_path) != True :
                print("\nDownloading blob to \n\t" + download_file_path)
                with open(download_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob(blob).readall())
                
    except Exception as ex:
        print('Exception:')
        print(ex)

TIME_TO_SLEEP = 5 * 60

def getLatestFile():
    list_of_files = glob.glob(local_path+'*RM19*.jpeg')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file

    

def downloadAndProcessLatest():
    while True:
        downloadLatest()
        latestFile = getLatestFile()
        image = load_image_cv2(latestFile)
        img_mod_ar = np.expand_dims(image,axis = 0)
        results = model.detect(img_mod_ar, verbose = 0)
        r = results[0]
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        msg = "Cars found in image:" + str(len(car_boxes))
        print(latestFile)
        print(msg)

        # Draw each box on the frame
        for box in car_boxes:
            # print("Car: ", box)

            y1, x1, y2, x2 = box

            # Draw the box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        time_now = time.ctime().replace(' ', '_')
        car_count = str(len(car_boxes))
        cv2.imwrite(f'processed/processed_{time_now}_{car_count}.jpeg',image)
        time.sleep(TIME_TO_SLEEP)
        
        
if __name__ == '__main__' and len(sys.argv) > 1:
    if(sys.argv[1] == 'downloadAll'):
        downloadAll()
    elif sys.argv[1] == 'downloadAndProcessLatest':
        downloadAndProcessLatest()
    else:
        println('Invalid Option')
else:
    print(__name__)
    print(sys.argv)
    print(sys.argv[1])
    
