from flask import Flask, request, redirect, jsonify
import glob
import os
from flask import send_file
from PIL import Image

from flask_cors import CORS

__name__ = 'parkinglotcet'

app = Flask(__name__)

CORS(app)

image_folder = 'processed/'

@app.route("/")
def hello():
    return "Hello World"

def getLatestFile():
    list_of_files = glob.glob(image_folder+'*.jpeg')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file


#@app.route('/parkinglot', methods = ['GET'])
#def parkinglot():
#    latestFile =  getLatestFile()
#    return latestFile[latestFile.rfind('_')+1 : latestFile.find('.jpeg')] + ' cars parked'

@app.route('/parkinglot', methods = ['GET'])
def parkinglotJson():
    latestFile =  getLatestFile()
    carCount = int(latestFile[latestFile.rfind('_')+1 : latestFile.find('.jpeg')])
    timestamp =latestFile[latestFile.find('_')+1 : latestFile.rfind('_')]
    responseDict = {'timestamp':timestamp, 'carCount': carCount}
    return jsonify(responseDict)

@app.route('/latestimage', methods = ['GET'])
def latestImage():
    print('latest file: ', getLatestFile())
    return send_file(getLatestFile(), mimetype='image/jpeg')
