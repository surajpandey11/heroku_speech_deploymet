# IMPORT NECESSARY LIBRARIES
from flask import Flask,jsonify,request,render_template
import librosa 
import numpy as np
import tensorflow as tf
import pandas as pd
import os # interface with underlying OS that python is running on
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.ker
import glob
import re
# Keras
from keras.models import load_model
# Flask utils
from flask import redirect, url_for, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



# instanciate object for Flask
app = Flask(__name__)


# Path foor model 1
MODEL_1_PATH = 'models/2nd_model.h5'

# Path for model 2
MODEL_2_PATH = 'models\model_2d_mfcc.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_1_PATH)
model2=tf.keras.models.load_model(MODEL_2_PATH)
          

print('Model loaded. Check http://127.0.0.1:5000/')

# function for model_2 predictions
def model_2_prediction(file_path,model2):
    n_mfcc = 30
    sampling_rate=44100
    n=n_mfcc
    f={'file_path':[file_path]}
    df= pd.DataFrame(f)
    X = np.empty(shape=(df.shape[0],n , 216, 1))
    input_length = 44100 * 2.5
    cnt=0
    data, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)

    if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

    
    
    
    n_mfcc = 30
    
    MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
    MFCC = np.expand_dims(MFCC, axis=-1)
    X[cnt,] = MFCC
    
    

    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean)/std

    prediction_model_2 = model2.predict(X)

    prediction_model_2=prediction_model_2.argmax(axis=1)
    prediction_model_2 = prediction_model_2.astype(int).flatten()
   

    return prediction_model_2

# Function for model_1 prediction
def model_predict(path, model):

    counter=0 

    df = pd.DataFrame(columns=['mel_spectrogram'])
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    
    #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    #temporally average spectrogram
    log_spectrogram = np.mean(db_spec, axis = 0)
    df.loc[counter] = [log_spectrogram]
    counter=counter+1 

    df_combined=pd.DataFrame(df['mel_spectrogram'].values.tolist())

    df_combined = df_combined.fillna(0)

    X_test = np.array(df_combined)

    X_test = X_test[:,:,np.newaxis]
 

    predict = model.predict(X_test)
    predict=predict.argmax(axis=1)
    predict = predict.astype(int).flatten()
    
    return predict

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        audio_file = request.files['file']
        
        

        # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(audio_file.filename))
        # audio_file.save(file_path)

        #  Make prediction from model_1
        preds = model_predict(audio_file, model)

        # Make prediction from model_2
        # preds2= model_2_prediction(audio_file,model2)

        
        return render_template('show.html', data=(preds))

#   return none
if __name__=="__main__":
    app.run(debug=True)