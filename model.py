import pandas as pd
import numpy as np
import os
import cv2
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
import matplotlib.pyplot as plt

DATA_DIR = '/opt/carnd_p3/data'
USE_AUGMENTATION = True
USE_LEFT_RIGHT = True

driving_log = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"), skipinitialspace=True)
# print(driving_log.head())

driving_log["center"] = '/opt/carnd_p3/data/' + driving_log["center"]
driving_log["left"] = '/opt/carnd_p3/data/' + driving_log["left"]
driving_log["right"] = '/opt/carnd_p3/data/' + driving_log["right"]
print(driving_log.head())

images = []
for image in driving_log["center"]:
    images.append(cv2.imread(image))

if USE_LEFT_RIGHT:
    for image in driving_log["left"]:
        images.append(cv2.imread(image))

    for image in driving_log["right"]:
        images.append(cv2.imread(image))

images = np.array(images)
print("Image shape: ", images.shape)

if USE_AUGMENTATION:
    images_flipped = images[:, :, ::-1, :]
    images = np.concatenate((images, images_flipped))
    print("Image shape after adding flips: ", images.shape)

X_train = np.array(images)
y_train_center = np.array(driving_log["steering"].values)

if USE_LEFT_RIGHT is True:
    correction = 0.1  # this is a parameter to tune
    y_train_left = np.array(driving_log["steering"].values) + correction
    y_train_right = np.array(driving_log["steering"].values) - correction
    y_train = np.concatenate((y_train_center, y_train_left, y_train_right), axis=0)
else:
    y_train = y_train_center

if USE_AUGMENTATION is True:
    y_train_flipped = -y_train
    y_train = np.concatenate((y_train, y_train_flipped))

# print(y_train)
# print(y_train.min())
# print(y_train.max())

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save("model.h5")

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("loss.png")
# plt.show()
