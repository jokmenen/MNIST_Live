## IMPORTS
##########
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from PIL import Image, ImageFile
from base64 import b64decode
from io import BytesIO

import numpy
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.backend import clear_session
from sklearn.preprocessing import StandardScaler
##########

app = Flask(__name__) #init Flask

clear_session() #clear any previous keras sessions
graph = tf.get_default_graph() #Set default graph, nessecary for working with tf in Flask because of multithreading

## LOADING MODEL
################
with open("mnist.json") as json:
    model_json = json.read()

model = model_from_json(model_json)
model.load_weights("mnist.h5")
opt = Adam(lr = 0.01)
model.compile(optimizer=opt, loss = "categorical_crossentropy")
################


## ROUTING
##########
@app.after_request
def add_header(response):
    # response.cache_control.max_age = 10
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/')
def main():
    return render_template('base.html')

@app.route('/sendImage', methods=["POST"])
def processImage():


    img64 = request.data.decode('utf-8') # Retrieve img from AJAX request

    # Split & Decode the header so it can be loaded
    header, encoded = img64.split(",", 1)
    img_b = b64decode(encoded)

    #Load the image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(BytesIO(img_b))
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((8,8), Image.NEAREST) # Resize to 8*8 with Nearest resampling. Nearest seemed the best method.
    image.save('img2.png', 'PNG') #DEBUG

    np_im = numpy.array(image) #Transform into numpy array for doing the prediction
    #np_im = numpy.array(image)
    #np_im = (np_im*-1) + 255
    #np_im = np_im.astype("float32")

    #scaler = StandardScaler()
    #np_im = scaler.fit_transform(np_im)
    np_im = ((np_im*-1) + 255)/255*16 #Invert colors and scale to 4 bit pixels

    #np_im = numpy.dot(np_im,16)

    #DEBUG
    numpy.save("prereshape.npy",np_im)
    new_im = Image.fromarray(np_im).convert("RGBA")
    new_im.save("numpy_img2.png")
    #/DEBUG

    #Reshape for use in model
    np_im = np_im.reshape((1,64))
    np_im = np_im.astype("int").astype("float32") # Rounding Cheeze
    #return str(np_im.shape)

    # This graph stuff is again needed for running tf in Flask
    global graph
    with graph.as_default():
        return "Prediction: {}".format(*model.predict_classes(np_im)) # Return the prediction to the Ajax request


########
if __name__ == '__main__':
    app.run()
