
import os
import math
import itertools
import time

from typing import Tuple, List
from pathlib import Path

import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
import pandas as pd
import streamlit as st
from mrcnn.model import MaskRCNN
from PIL import ImageOps
from keras import backend as K

import utils


IMAGE_PATH = 'images'

# Root directory of the project
ROOT_DIR = Path(".")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


### Functions ###

@st.cache(allow_output_mutation=True)
def keras_setup():
    ### Limit GPU RAM initially grabbed by TensorFlow
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session  
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
    # tf_config.gpu_options.per_process_gpu_memory_fraction=0.333
    sess = tf.Session(config=tf_config)  
    set_session(sess)  # set this TensorFlow session as the default session for Keras.
    return sess


@st.cache(allow_output_mutation=True)
def get_predicted_images_dict():
    return dict()


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)
    return np.array(car_boxes)


@st.cache(allow_output_mutation=True)
def get_image_loader():
    return utils.ImageLoader(path=IMAGE_PATH)


@st.cache(allow_output_mutation=True)
def get_model():
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model 


account_url = os.getenv('ACCOUNT_URL')
container_name = os.getenv('CONTAINER_NAME')
sas_token = os.getenv('SAS_TOKEN')

@st.cache(allow_output_mutation=True)
def get_blob_downloader():

    try:
        blob_dl = utils.BlobDownloader(account_url=account_url,
                                container_name=container_name,
                                sas_token=sas_token)
        return blob_dl
    except Exception as e:
        print('BlobDownloader not created')
        print(e)
    

account_name = os.getenv('ACCOUNT_NAME')
sas_token_table = os.getenv('SAS_TOKEN_TABLE')
table_name = os.getenv('TABLE_NAME')

@st.cache(allow_output_mutation=True)
def get_table_saver():
    try:
        ts = utils.TableSaver(account_name=account_name,
                              sas_token=sas_token_table,
                              table_name=table_name)
        return ts
    except Exception as e:
        print('TableSaver now created')
        print(e)


def get_missing_image_list(image_loader, blob_downloader) -> List[str]:
    image_loader.reset_image_list()
    local_imgs = {str(Path(fn).name) for fn in image_loader.get_image_list()}
    cloud_imgs = blob_downloader.get_daytime_image_list()
    missing_imgs = set(cloud_imgs) - local_imgs
    return missing_imgs, len(local_imgs), len(cloud_imgs)


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def adjust_brightness_and_contrast(img, brightness, contrast):
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def predict_on_image(model, image) -> Tuple[List, str]:
    """Use `model` to predict cars on `image`.

    Expects a single image as a Numpy array.

    Returns
    -------
    - List of bounding boxes 
    - Image with boxes draw on it
    """
    # Add dimension to single image to match model's input.
    img_mod_ar = np.expand_dims(img_mod, axis=0)
    # Predict with model.
    results = model.detect(img_mod_ar, verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    # Filter the results to only grab the car / truck bounding boxes
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    # msg = "Cars found in image:" + str(len(car_boxes))

    return_img = img_mod.copy()

    # Draw each box on the frame
    for box in car_boxes:
        # print("Car: ", box)

        y1, x1, y2, x2 = box

        # Draw the box
        cv2.rectangle(return_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return car_boxes, return_img


###################################
### Start main script logic ###
###################################

# This must be called before other TF/Keras functions
sess = keras_setup()

# Set up caching
image_loader = get_image_loader()
predict_cache = get_predicted_images_dict()

# Load the only model we have right now.
# This will change when we want to add more models.
model = get_model()

# Without this, TF throws an error about a layer not being in graph.
# It seems repetative with `set_session(sess)` from above, but it 
# has to be executed after the model is loaded.
K.set_session(sess)  

# Attempt to create blob downloader
blob_downloader = get_blob_downloader()

# Attempt to create table saver
table_saver = get_table_saver()

# Model selection widgets
st.sidebar.selectbox('Choose model', ['MaskRNN'])
st.sidebar.markdown('---')

# App title
st.title('Parking Lot app')

# Image list filters
if st.sidebar.checkbox('Show Filter Options'):
    # Room filter
    image_filter_rooms = st.sidebar.multiselect('Choose rooms', ('RM19', 'pike'), default='RM19')
    # Date filter
    image_filter_date_start = st.sidebar.date_input('Choose start date')
    image_filter_date_end = st.sidebar.date_input('Choose end date')
else:
    image_filter_rooms = None
    image_filter_date_start = None
    image_filter_date_end = None

# Load image name list
images = image_loader.get_image_list(rooms=image_filter_rooms,
                                    start_date=image_filter_date_start,
                                    end_date=image_filter_date_end)

# Create slider for choosing image index
num_of_images = len(images)
img_path_index = st.sidebar.slider('Choose image', min_value=0, max_value=num_of_images-1)
img_path = images[img_path_index]


st.sidebar.markdown('---')

# Load image
img_orig = image_loader.get_image(img_path)
img_mod = img_orig.copy()

# Show original
st.write(f'Filename: {img_path}')
if st.sidebar.checkbox('Display original image', value=True):
    st.image(img_orig, caption="Original")

total_cars_in_original = -1
if st.sidebar.checkbox('Predict on original (cached)'):
    if img_path in predict_cache:
        img_orig_pred, total_cars_in_original = predict_cache[img_path]
    else:
        # Predict on original image
        car_boxes, img_orig_pred = predict_on_image(model, img_orig)
        total_cars_in_original = len(car_boxes)
        predict_cache[img_path] = (img_orig_pred, total_cars_in_original)

    msg = "Cars found in original image:" + str(total_cars_in_original)
    print(msg)
    # st.write(msg)
    st.image(img_orig_pred, msg)   

st.sidebar.markdown('---')

# Color manipulation
color_adjusted = False
if st.sidebar.checkbox('Show Color Options'):
    img_mod = white_balance(img_mod)
    brightness = st.sidebar.slider('Choose brightness', min_value=-127, max_value=127, value=0)
    contrast = st.sidebar.slider('Choose contrast', min_value=-127, max_value=127, value=0)
    img_mod = adjust_brightness_and_contrast(img_mod, brightness, contrast)
    color_adjusted = True
    # img_mod = img_mod.copy()
    # st.image(img_mod, caption="Color adjusted")


# define slices
sliced = False
if st.sidebar.checkbox('Show Slicing Options'):
    # img_mod = img_mod.copy()
    num_submats_x = st.sidebar.slider('Choose X Slices', min_value=1, max_value=10, value=4)
    num_submats_y = st.sidebar.slider('Choose Y Slices', min_value=1, max_value=10, value=2)

    # slice image
    height, width, channels = img_mod.shape
    submat_w = math.floor(width/num_submats_x)
    submat_h = math.floor(height/num_submats_y)
    offsets = []

    submats = np.array([])
    for i in range(num_submats_x):
        for j in range(num_submats_y):

            x_min = i * submat_w
            x_max = (1 + i) * submat_w
            y_min = j * submat_h
            y_max = (1 + j) * submat_h

            offsets.append((x_min, x_max, y_min, y_max))

            submat = np.array(img_mod[y_min:y_max, x_min:x_max])

            # draw slices
            cv2.rectangle(img_mod, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            if len(submats) is 0:
                submats = [submat]
            else:
                submats = np.concatenate((submats, [submat]), axis=0)
    sliced = True
    # Display image with slices
    # st.image(img_mod, caption="Sliced Image")

if color_adjusted or sliced:
    adjustments = itertools.compress(['Color adjusted', 'Sliced'],
                                    [color_adjusted, sliced])
    st.image(img_mod, caption=f"{', '.join(adjustments)} image")


### Prediction on modified image
if st.button("I'm Ready. Let's go!"):

    if sliced:
        total_images = num_submats_x * num_submats_y
        latest_iteration = st.empty()
        latest_iteration.text(f'Processing Slice 0/{total_images}')
        bar = st.progress(0)
        total_cars_found = 0
        step = 0

        for i in range(num_submats_x):
            for j in range(num_submats_y):

                with sess.as_default():
                    with sess.graph.as_default():
                        # st.write(submats.shape)
                        submat_ar = np.expand_dims(submats[step], axis=0)
                        submat_result = model.detect(submat_ar, verbose=0)

                # Mask R-CNN assumes we are running detection on multiple images.
                # We only passed in one image to detect, so only grab the first result.
                r = submat_result[0]

                # The r variable will now have the results of detection:
                # - r['rois'] are the bounding box of each detected object
                # - r['class_ids'] are the class id (type) of each detected object
                # - r['scores'] are the confidence scores for each detection
                # - r['masks'] are the object masks for each detected object (which gives you the object outline)

                # Filter the results to only grab the car / truck bounding boxes
                car_boxes = get_car_boxes(r['rois'], r['class_ids'])
                total_cars_found += len(car_boxes)
                x_min, x_max, y_min, y_max = offsets[step]

                # Draw each box on the frame
                for box in car_boxes:
                    # print("Car: ", box)

                    y1, x1, y2, x2 = box

                    # Draw the box
                    cv2.rectangle(img_mod, (x1 + x_min, y1 + y_min), (x2 + x_min, y2 + y_min), (0, 255, 0), 1)

                step += 1
                perc_progress = math.floor(step/total_images * 100)
                latest_iteration.text(f'Processing Slice {step}/{total_images}')
                bar.progress(perc_progress)
                time.sleep(0.1)
    else:
        # Predict on original image
        car_boxes, img_mod = predict_on_image(model, img_mod)
        total_cars_found = len(car_boxes)

    msg = f"Cars found in modified image: {total_cars_found} (original: {total_cars_in_original if total_cars_in_original > -1 else '???'})"
    st.image(img_mod, msg) 


st.sidebar.markdown('---')
if st.sidebar.checkbox('Manual Count'):
    st.sidebar.markdown('If you have manually counted the cars, you can save that data here.')
    manual_car_count = st.sidebar.text_input('Manual car count')
    if st.sidebar.button('Save manual car count'):
        table_saver.save_count(img_path, manual_car_count)


st.sidebar.markdown('---')

###  Check image list
if st.sidebar.checkbox("Check for additional images."):
    text_ph = st.sidebar.empty()
    missing_images, n_local, n_cloud = get_missing_image_list(image_loader, blob_downloader)
    if len(missing_images) > 0:
        if st.sidebar.button(f'Click to download missing images.'):
            blob_downloader.download_images(missing_images, IMAGE_PATH)
    missing_images, n_local, n_cloud = get_missing_image_list(image_loader, blob_downloader)
    text_ph.text(f'{n_local}/{n_cloud} images available locally.')