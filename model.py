import pandas as pd
import numpy as np
import os
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
import matplotlib.pyplot as plt

DATA_DIR = '/opt/carnd_p3/data'
VALIDATION_SIZE = 0.1
BATCH_SIZE = 32
DROPOUT = 0.2
N_EPOCHS = 3
SEED = 42

# if using both, this will lead to memory error if not using a Generator/Dataloader
# ATTENTION: flips will lead to leakage in validation score if splitting is not done right
# splitting must be done before the augmentation
# and also before adding left and right pictures to the mix
USE_AUGMENTATION = True
USE_LEFT_RIGHT = True
# angles are normalized from +-25 deg to +-1
# 0.15 corresponds to 3.75 deg
# 0.20 corresponds to 5.00 deg
LEFT_RIGHT_ANGLE_CORRECTION = 0.2

DROP_ZERO_STEERINGS = 0.30  # set fraction to keep

driving_log = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"), skipinitialspace=True)
# print(driving_log.head())

driving_log["center"] = '/opt/carnd_p3/data/' + driving_log["center"]
driving_log["left"] = '/opt/carnd_p3/data/' + driving_log["left"]
driving_log["right"] = '/opt/carnd_p3/data/' + driving_log["right"]
print("Driving Log head: ", driving_log.head())

train_driving_log, validation_driving_log = train_test_split(driving_log, test_size=VALIDATION_SIZE, random_state=SEED)
print("Len train_driving_log: ", len(train_driving_log))
print("Len validation_driving_log: ", len(validation_driving_log))

# plot an histogramm of the training steering values
plt.hist(train_driving_log["steering"], bins=64)
plt.title('Histogramm of Steering value')
plt.ylabel('Distribution')
plt.xlabel('Steering value')
plt.savefig("steering_hist.png")
plt.close()

# The majority of the samples has zero steering (seems to be recorded with a keyboard instead of mouse)
# We will either remove those to emphasize more on turning (and to reduce training time)
# or do some label noise to reduce impact on augmentation (+- LEFT_RIGHT_ANGLE_CORRECTION)
if DROP_ZERO_STEERINGS > 0:
    train_driving_log = train_driving_log.loc[train_driving_log["steering"] != 0].append(train_driving_log[train_driving_log['steering'] == 0].sample(frac=DROP_ZERO_STEERINGS, random_state=SEED))

    # plot an histogramm of the training steering values
    plt.hist(train_driving_log["steering"], bins=64)
    plt.title('Histogramm of Steering value')
    plt.ylabel('Distribution')
    plt.xlabel('Steering value')
    plt.savefig("steering_hist_after.png")
    plt.close()


def generator(samples_df, batch_size=32, is_training=False):
    num_samples = len(samples_df)

    while 1:  # Loop forever so the generator never terminates
        # if training: shuffle dataframe on epoch level, for validation set shuffling should not be used!
        if is_training is True:
            samples_df = samples_df.sample(frac=1).reset_index(drop=True)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_df[offset:offset + batch_size]
            images = []
            angles = []
            for index, row in batch_samples.iterrows():
                # cv2 reads in BGR by default, we need RGB as drive.py uses PIL Image which is RGB
                image_center = cv2.cvtColor(cv2.imread(row["center"]), cv2.COLOR_BGR2RGB)
                images.append(image_center)
                angles.append(row["steering"])
                if (USE_AUGMENTATION is True) and (is_training is True):
                    image_center_flipped = np.fliplr(image_center)
                    images.append(image_center_flipped)
                    angles.append(-row["steering"])
                # Use the left and right pictures if in training set
                if (USE_LEFT_RIGHT is True) and (is_training is True):
                    image_left = cv2.cvtColor(cv2.imread(row["left"]), cv2.COLOR_BGR2RGB)
                    image_right = cv2.cvtColor(cv2.imread(row["right"]), cv2.COLOR_BGR2RGB)
                    images.append(image_left)
                    angles.append(row["steering"] + LEFT_RIGHT_ANGLE_CORRECTION)
                    images.append(image_right)
                    angles.append(row["steering"] - LEFT_RIGHT_ANGLE_CORRECTION)
                    if (USE_AUGMENTATION is True) and (is_training is True):
                        image_left_flipped = np.fliplr(image_left)
                        image_right_flipped = np.fliplr(image_right)
                        images.append(image_left_flipped)
                        angles.append(-(row["steering"] + LEFT_RIGHT_ANGLE_CORRECTION))
                        images.append(image_right_flipped)
                        angles.append(-(row["steering"] - LEFT_RIGHT_ANGLE_CORRECTION))

            X_train = np.array(images)
            y_train = np.array(angles)

            # shuffle again on batch level (needed?)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_driving_log, batch_size=BATCH_SIZE, is_training=True)
validation_generator = generator(validation_driving_log, batch_size=BATCH_SIZE, is_training=False)

model = Sequential()
# range -0.5 to +0.5 as proposed in the lessons seems to be very narrow.
# I increased the range to +-1
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
# cropping 20 pixels from the bottom as proposed in the lessons seems to be a little bit too few
# the hood is still visible and may break the augmentation. Thus, I increased it to 25 pixels
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(DROPOUT))
model.add(Dense(100, activation="relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(50, activation="relu"))
# model.add(Dropout(DROPOUT))
model.add(Dense(10, activation="relu"))
# model.add(Dropout(DROPOUT))
model.add(Dense(1))
print(model.summary())

multiplier = 1
if USE_AUGMENTATION is True:
    multiplier *= 2
if USE_LEFT_RIGHT is True:
    multiplier *= 3
training_samples = int(len(train_driving_log) * multiplier)
validation_samples = int(len(validation_driving_log))

print("Training on", training_samples, "samples")
print("Validating on", validation_samples, "samples")

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=int(np.ceil(len(train_driving_log) / BATCH_SIZE)),
                                     validation_data=validation_generator,
                                     validation_steps=int(np.ceil(len(validation_driving_log) / BATCH_SIZE)),
                                     epochs=N_EPOCHS, verbose=1)

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
plt.close()
