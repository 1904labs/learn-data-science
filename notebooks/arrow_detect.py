import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras import backend as K


cap = cv2.VideoCapture(0)
model = load_model('mixed_model_val_100__no_opt.h5')
#graph = tf.get_default_graph()
print(K.image_data_format())
while(True):
    # Capture frame-by-frame
    ret, orig = cap.read()
    frame = orig
    frame =  frame.astype('float32')
    frame /= 255
    frame = cv2.resize(frame, (150,150))
    cv2.imshow('frame1',frame)
    frame = frame.reshape(1,150,150,3)
    # Our operations on the frame come here
    #with graph.as_default():
    keras_pred = model.predict_classes(frame)[0]
    print(model.predict(frame))

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('d'):
        cv2.imshow('frame',orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
