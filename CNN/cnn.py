from datetime import datetime

import pygal
import talos
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

import helpers
from helpers import INPUT_SHAPE

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

dataset_dir_name, checkpoint_dir_name, stats_dir_name, model_name = helpers.getInput()

data = helpers.calibrate(dataset_dir_name)

training_data, testing_data = helpers.getData(data, 75)

trainingImages = helpers.getMatrix(training_data)
testingImages = helpers.getMatrix(testing_data)

print("_"*40)
print("Image translation to matrix completed")
print("_"*40)

now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
# NAME = f"{model_name}_{date_time}"
tensorboard = TensorBoard(log_dir = f'logs/{model_name}')

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(INPUT_SHAPE)))
    model.add(Conv2D(96, 11, strides=4, activation='relu'))
    model.add(MaxPooling2D(3, 2))
    model.add(Conv2D(256, 5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3, 2))
    model.add(Conv2D(384, 3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(3, 2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='elu'))


    return model

    return model

def train_model(model, model_name, tensorboard):
    """
    Train the model
    """

    checkpoint = ModelCheckpoint('models/' + model_name + '/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    callbacks_list = [checkpoint, tensorboard]

    model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse', 'mae', 'mape', 'cosine'])

    history = model.fit(trainingImages, training_data['Steering Angles'].values,
                    batch_size=64,
                    epochs=10,
                    validation_data=(testingImages, testing_data['Steering Angles'].values),
                    callbacks=callbacks_list, verbose=1)
    model.summary()
    return history

if __name__ == '__main__':
    model = build_model()
    history = train_model(model, model_name, tensorboard)
