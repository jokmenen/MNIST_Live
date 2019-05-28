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


app = Flask(__name__)

clear_session()
graph = tf.get_default_graph()

#REMOVE
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
digits = load_digits()

X_train, X_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size = 1/3, random_state = 1)

onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)

#/REMOVE


with open("mnist.json") as json:
    model_json = json.read()

model = model_from_json(model_json)
model.load_weights("mnist.h5")
opt = Adam(lr = 0.01)
model.compile(optimizer=opt, loss = "categorical_crossentropy")

print(digits.data[1].shape)

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
    img64 = request.data.decode('utf-8')
    header, encoded = img64.split(",", 1)
    img_b = b64decode(encoded)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(BytesIO(img_b))
    image = image.convert("L")
    image = image.resize((8,8), Image.NEAREST) #TODO: check of ik dit kan verbeteren
    image.save('img2.png', 'PNG')

    np_im = numpy.array(image)
    #np_im = numpy.array(image)
    #np_im = (np_im*-1) + 255
    #np_im = np_im.astype("float32")

    #scaler = StandardScaler()
    #np_im = scaler.fit_transform(np_im)
    np_im = ((np_im*-1) + 255)/255*16

    #np_im = numpy.dot(np_im,16)
    numpy.save("prereshape.npy",np_im)
    new_im = Image.fromarray(np_im).convert("RGBA")
    new_im.save("numpy_img2.png")
    np_im = np_im.reshape((1,64))
    np_im = np_im.astype("int").astype("float32")
    print(np_im)
    #return str(np_im.shape)


    global graph
    with graph.as_default():
        return "Prediction: {}".format(*model.predict_classes(np_im))

if __name__ == '__main__':
    app.run()