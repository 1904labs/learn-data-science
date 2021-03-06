import pyaudio
from queue import Queue
from threading import Thread
import sys
import time
import numpy as np
from pyaudio import *
from  scipy import signal
import matplotlib.mlab as mlab
from tensorflow.keras.models import Model, load_model, Sequential
#import switch

chunk_duration = 0.5
fs = 44100 #
Tx = 5511
model = load_model('tr_model.h5')

chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)



def detect_triggerword_spectrum(x):
    """
    Function to predict the location of the trigger word.
    
    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    # the spectogram outputs  and we want (Tx, freqs) to input into the model
    #print(x.shape)
    #x = np.pad(x,Tx - x.shape[1])
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    #print(x.shape)
    predictions = model.predict(x)
    return predictions.reshape(-1)

def get_spectrogram(data):
    """
    Function to compute a spectrogram.
    
    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    #print(data.shape)
    
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    #print(data.ndim)
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=2,
        stream_callback=callback)
    return stream


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.2):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.
    
    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False





# Queue to communiate between the audio callback and main thread
q = Queue()
#feed_samples = 10 * 1000
run = True

silence_threshold = 150

# Run the demo for a timeout seconds
timeout = time.time() + 0.5*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold    
    #if time.time() > timeout:
    #    run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)


p = pyaudio.PyAudio()

"""
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                stream_callback=callback)
"""
stream = get_audio_input_stream(callback)
stream.start_stream()
#import subprocess


try:
    while run:
        data = q.get()
        #subprocess.call(['aplay' , 'test.wav'])
        spectrum = get_spectrogram(data)
        preds = detect_triggerword_spectrum(spectrum)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        print(new_trigger)
        if new_trigger:
            #switch.toggleLed()
            sys.stdout.write('1')
            sys.stdout.flush()
except (KeyboardInterrupt, SystemExit):
    #print('error')
    #pass
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False
        
stream.stop_stream()
stream.close()
