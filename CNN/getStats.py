import random as rd
from time import time

import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

import helpers

dataset_dir_name, checkpoint_dir_name, stats_dir_name, model_name = helpers.getInput()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

data = helpers.calibrate(dataset_dir_name)

training_data, testing_data = helpers.getData(data, 75)

trainingImages = helpers.getMatrix(training_data)
testingImages = helpers.getMatrix(testing_data)

print("_"*40)
print("Image translation to matrix completed")
print("_"*40)

model = load_model("models/AlexNetAdapted/model-010.h5")
Y_val = testing_data['Steering Angles'].values
X_val = testingImages
total_time = []
all_results = []
for i in range(15000):
    random = rd.randint(0, 17000)
    t1 = time()
    result = model.predict(np.expand_dims(X_val[random], 0))
    t2 = time()
    all_results.append(result)
    total_time.append(t2 - t1)

print("The average prediction time is: " + str(sum(total_time)/len(total_time)))
