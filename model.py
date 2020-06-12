import pandas as pd
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

DATA_DIR = '/opt/carnd_p3/data'

driving_log = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"))
# print(driving_log.head())

driving_log["center"] = '/opt/carnd_p3/data/' + driving_log["center"]
print(driving_log.head())

images = []
for image in driving_log["center"]:
    images.append(cv2.imread(image))

X_train = np.array(images)
y_train = np.array(driving_log["steering"].values)

# print(y_train)
# print(y_train.min())
# print(y_train.max())

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save("model.h5")
