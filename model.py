import numpy
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import h5py

digits = load_digits()

X_train, X_val, y_train, y_val = train_test_split(digits.data, digits.target, test_size = 1/3, random_state = 1)

onehot = LabelBinarizer()
Y_train = onehot.fit_transform(y_train)
Y_val   = onehot.transform(y_val)

model = Sequential()

model.add(Dense(64, input_dim = 64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

opt = Adam(lr = 0.01)
model.compile(optimizer=opt, loss = "categorical_crossentropy")
model.fit(X_train,Y_train, epochs = 100)


# serialize model to JSON & save weights to h5
model_json = model.to_json()
with open("mnist.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mnist.h5")


with open("mnist.json") as json:
    model_json = json.read()

modelnew = model_from_json(model_json)
modelnew.load_weights("mnist.h5")
opt = Adam(lr = 0.01)
modelnew.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=['accuracy'])
score = modelnew.evaluate(X_val, Y_val, verbose=0)
print(score)
