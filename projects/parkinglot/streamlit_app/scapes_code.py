# pylint: disable=import-error
import streamlit as st
import os
import math
import time
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import pandas as pd
from PIL import ImageOps
from keras.preprocessing import image
from keras import backend as K

@st.cache(allow_output_mutation=True)
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

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
    
@st.cache(allow_output_mutation=True)
def get_model():
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model 

model = get_model()
K.set_session(sess)  # Without this, TF throws an error about a layer not being in graph.

# def load_image_skimage():
#     img_pil = image.load_img(img_path, target_size = None)
#     img_mod = img_pil
#     img_mod = ImageOps.flip(img_mod)
#     # st.write(img_pil.mode)
#     # img_kev_mod = img_kev_pil.crop((230, 80, 520, 420))
#     # img_mod = img_pil.resize((128, 128))

#     # img_mod = image.img_to_array(img_mod) / 255
#     img_mod = image.img_to_array(img_mod)


def load_model_cv2(fn):
    """Load image file using OpenCV"""
    img = cv2.imread(str(fn), 1)
    img = img[:, :, ::-1]
    img = cv2.flip(img, 0)
    return img

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

st.sidebar.selectbox('Choose model', ['MaskRNN'])

# Location of parking spaces
parked_car_boxes = None

# App title
st.title('Parking Lot app')

# Load image name list
image_path = Path('images')
images = [str(fn) for fn in image_path.iterdir() if 'DS_Store' not in str(fn)]

# Create slider for choosing image index
num_of_images = len(images)
img_path_index = st.sidebar.slider('Choose image', min_value=0, max_value=num_of_images-1)
img_path = images[img_path_index]
st.write(f'Filename: {img_path}')

# Switch to cv2
# contrast, hue, saturation, hsv 

# Load image
img_mod = load_model_cv2(img_path)

# Color manipulation
img_mod = white_balance(img_mod)
brightness = st.sidebar.slider('Choose brightness', min_value=-127, max_value=127, value=0)
contrast = st.sidebar.slider('Choose contrast', min_value=-127, max_value=127, value=0)
img_mod = adjust_brightness_and_contrast(img_mod, brightness, contrast)
preview_image = img_mod.copy()

# define slices
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
        cv2.rectangle(preview_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        if len(submats) is 0:
            submats = [submat]
        else:
            submats = np.concatenate((submats, [submat]), axis=0)

# Display image with slices
st.image(preview_image, caption="Sliced Image")

if st.button("I'm Ready. Let's go!"):

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
            
    # with sess.as_default():
    #     with sess.graph.as_default():
    #         # st.write(img_mod.shape)
    #         img_mod_ar = np.expand_dims(img_mod, axis=0)
    #         results = model.detect(img_mod_ar, verbose=0)


    msg = "Cars found in image:" + str(total_cars_found)
    print(msg)
    st.write(msg)

    st.image(img_mod)  