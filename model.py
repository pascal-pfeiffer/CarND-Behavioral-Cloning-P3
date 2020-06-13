import pandas as pd
import numpy as np
import os
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
import matplotlib.pyplot as plt

DATA_DIR = '/opt/carnd_p3/data'
VALIDATION_SIZE = 0.2
BATCH_SIZE = 32

# if using both, this will lead to memory error if not using a Generator/Dataloader
# ATTENTION: flips will lead to leakage in validation score if splitting is not done right
# splitting must be done before the augmentation
# and also before adding left and right pictures to the mix
USE_AUGMENTATION = True
USE_LEFT_RIGHT = False
LEFT_RIGHT_ANGLE_CORRECTION = 0.06

driving_log = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"), skipinitialspace=True)
# print(driving_log.head())

driving_log["center"] = '/opt/carnd_p3/data/' + driving_log["center"]
driving_log["left"] = '/opt/carnd_p3/data/' + driving_log["left"]
driving_log["right"] = '/opt/carnd_p3/data/' + driving_log["right"]
print("Driving Log head: ", driving_log.head())

train_driving_log, validation_driving_log = train_test_split(driving_log, test_size=VALIDATION_SIZE)
print("Len train_driving_log: ", len(train_driving_log))
print("Len validation_driving_log: ", len(validation_driving_log))


def generator(samples_df, batch_size=32, is_training=False):
    num_samples = len(samples_df)

    while 1:  # Loop forever so the generator never terminates
        # if training: shuffle dataframe on epoch level, for validation set shuffling should not be used!
        if is_training is True:
            samples_df = samples_df.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_df[offset:offset + batch_size]
            images = []
            angles = []
            for index, row in batch_samples.iterrows():
                image_center = cv2.imread(row["center"])
                images.append(image_center)
                angles.append(row["steering"])
                if (USE_AUGMENTATION is True) and (is_training is False):
                    image_center_flipped = np.fliplr(image_center)
                    images.append(image_center_flipped)
                    angles.append(-row["steering"])
                # Use the left and right pictures if in training set
                if (USE_LEFT_RIGHT is True) and (is_training is False):
                    image_left = cv2.imread(row["left"])
                    image_right = cv2.imread(row["right"])
                    images.append(image_left)
                    angles.append(row["steering"] + LEFT_RIGHT_ANGLE_CORRECTION)
                    images.append(image_right)
                    angles.append(row["steering"] - LEFT_RIGHT_ANGLE_CORRECTION)
                    if (USE_AUGMENTATION is True) and (is_training is False):
                        image_left_flipped = np.fliplr(image_left)
                        image_right_flipped = np.fliplr(image_right)
                        images.append(image_left_flipped)
                        angles.append(row["steering"] + LEFT_RIGHT_ANGLE_CORRECTION)
                        images.append(image_right_flipped)
                        angles.append(-(row["steering"] - LEFT_RIGHT_ANGLE_CORRECTION))

            X_train = np.array(images)
            y_train = np.array(angles)

            # shuffle again on batch level (needed?)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_driving_log, batch_size=BATCH_SIZE)
validation_generator = generator(validation_driving_log, batch_size=BATCH_SIZE)

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
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=np.ceil(len(train_driving_log) / BATCH_SIZE),
                                     validation_data=validation_generator,
                                     validation_steps=np.ceil(len(validation_driving_log) / BATCH_SIZE),
                                     epochs=5, verbose=1)

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
