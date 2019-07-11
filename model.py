import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras
import h5py
from PIL import Image, ImageFile

#TODO: MNIST uses 28x28, sklearn 8*8. Model decide what resolution canvas outputs through settings file

# input_x,input_y = None,None
input_x, input_y = 28, 28  # Get size of img to feed to model
#I made functions to make it easier to switch datasets
def sklearnDigits():
    print("Loading sklearn Digits")
    digits = load_digits()
    input_x,input_y = 8,8 #hier stond eerst 84 maar dat klopt vgm niet
    return train_test_split(digits.data, digits.target, test_size = 1/3, random_state = 1)

def mnistDigits():
    print("Loading MNIST Digits")
    input_x, input_y = 28,28 # Get size of img to feed to model

    train = np.loadtxt(open("MNIST\mnist_train.csv", "rb"), delimiter=",", skiprows=1)
    tr_indices = np.arange(len(train))
    train = train[np.random.choice(tr_indices, size = train.shape[0])]
    y_train, X_train = np.hsplit(train,[1])


    test = np.loadtxt(open("MNIST\mnist_train.csv", "rb"), delimiter=",", skiprows=1)
    te_indices = np.arange(len(test))
    test = test[np.random.choice(te_indices, size=test.shape[0])]
    y_val, X_val = np.hsplit(test, [1])



    return X_train, X_val, y_train, y_val

#Uncomment the line below to load sklearn load_digits data
#X_train, X_val, y_train, y_val = sklearnDigits()

#Load Mnist digits
X_train, X_val, y_train, y_val = mnistDigits()



print("Digits loaded")


# onehot = LabelBinarizer()
# Y_train = onehot.fit_transform(y_train)
# Y_val   = onehot.transform(y_val)

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255

X_train = X_train.reshape(X_train.shape[0],input_x,input_y,1)
X_val = X_val.reshape(X_val.shape[0],input_x,input_y,1)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape =  (input_x, input_y, 1)))
model.add(Conv2D(62, (3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(X_val, y_val))
#

#
#
# model.add(Dense(64, input_dim = input_dim, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
#
# model.add(Dense(10, activation = 'softmax'))
#
# opt = Adam(lr = 0.01)
# model.compile(optimizer=opt, loss = "categorical_crossentropy")
#model.fit(X_train,Y_train, epochs = 100, batch_size= 64)


# serialize model to JSON & save weights to h5
model_json = model.to_json()
with open("mnist.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mnist.h5")


with open("mnist.json") as json:
    model_json = json.read()

modelnew = model_from_json(model_json)
modelnew.load_weights("mnist.h5")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score = modelnew.evaluate(X_val, y_val, verbose=0)
print(score)

#output an example image file to debug the model implementation
new_im = Image.fromarray(X_train[3].reshape((28,28))).convert("RGBA")
new_im.save("model_image.png")
print(y_train[3])
