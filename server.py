from flask import Flask, request, jsonify
import os
import random
import time
import utils
import torch
import pdb
import numpy as np
from torch.utils.data import DataLoader
from utils import calculate_eer, step_decay, extract_all_feat
from tqdm import tqdm as tqdm
from hparam import hparam as hp
from data_load import VoxCeleb, VoxCeleb_utter
#from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
#from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, SILoss, get_centroids, \
#get_cossim, HybridLoss
from speech_embedder2 import SpeakerRecognition, SILoss
import wave
torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] =str(hp.device) # for multiple gpus
from get_embedding import get_embedding

logger = utils.init_logger('voice detection model')

app = Flask(__name__)

@app.route('/test', methods = ['POST'])
def test():
    if request.method == 'POST':
        req_json = request.json

        name = req_json['name']
        #answer = 'length of the req json is ' + str(len(req_json))
        return jsonify({'length of the req json': str(len(req_json)) })

@app.route('/health', methods = ['POST'])
def check():
    if request.method == 'POST':
        return jsonify({'response':'OK!'})

def get_customer_voice_10_seconds(file):
    voice = AudioSegment.from_wav(file)
    new_voice = voice[0:10000]
    file = str(file) + '_10seconds.wav'
    new_voice.export(file, format='wav')


@app.route('/get_embedding', methods = ['POST'])
def show_embedding():
    if request.method == 'POST':

        file1 = request.files["file1"]
        if file1 == None:
            return jsonify({'response': 'BAD, no file given!'})
        else:

            embedding = get_embedding(file1)
            return jsonify({'response': str(embedding)})

@app.route('/process_all', methods = ['POST'])
def process_all():
    if request.method == 'POST':
        file_wav = request.files['file_wav']
        if file_wav == None:
            return jsonify({'response': 'BAD, no wav conversation file given!'})
        else:

            embedding = get_embedding(file_wav)
            return jsonify({'response': str(embedding)})


@app.route('/compare_voices', methods = ['POST'])
def compare_voices():
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    embedding1 = get_embedding(file1)
    embedding2 = get_embedding(file2)

    embedding1 = embedding1 / torch.norm(embedding1, dim=1).unsqueeze(1)
    embedding2 = embedding2 / torch.norm(embedding2, dim=1).unsqueeze(1)
    score = torch.dot(embedding1.squeeze(0), embedding2.squeeze(0)).item()
    print(score)
    answer = ''
    if score > 0.9:
        answer = 'pass '
    else:
        answer = 'no pass '
    score = float(score*100)
    score = '%.2f' % score
    score = str(score)
    answer += score
    return jsonify({'response': answer })


import logging
import pandas as pd
import numpy as np

FORMAT = '[%(asctime)s] [%(process)d] [%(levelname)7s] [%(filename)s:%(lineno)s]: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S %z'


def init_logger(name):
    # Formatters
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

    # Handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

       # Loggers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # root level
    logger.addHandler(console_handler)
    return logger

class ApiError(Exception):
    result = None
    errcode = None
    errtext = None

    def __init__(self, errcode, errtext):
        Exception.__init__(self)
        self.result = 'err'
        self.errcode = errcode
        self.errtext = errtext

    def get_error(self):
        return {'result': self.result, 'errcode': self.errcode, 'errtext': self.errtext}

    def __str__(self):
        return str(self.get_error())

if __name__ == '__main__':
    app.run(debug=True, port=9090)
