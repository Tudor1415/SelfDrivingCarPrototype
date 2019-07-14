import pandas as pd
import numpy as np
import imageio

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (120, 160, 3)

def getInput():
    dataset_dir_name = input("Please input the dataset directory path: ")
    checkpoint_dir_name = input("Please input the checkpoint directory path: ")
    stats_dir_name = input("Please input the statistics directory path if you want them: ")
    model_name = input("Please input the model name: ")
    model_name_default = "test"
    dataset_dir_name_default = "D:/selfDrivingCar/dataset/ProcessedData"
    checkpoint_dir_name_default = "D:/selfDrivingCar/CNN/checkpoints"
    stats_dir_name_default = "D:/selfDrivingCar/CNN/stats"

    if not dataset_dir_name:
        dataset_dir_name = dataset_dir_name_default

    if not checkpoint_dir_name:
        checkpoint_dir_name = checkpoint_dir_name_default

    if not stats_dir_name:
        stats_dir_name = stats_dir_name_default

    if not model_name:
        model_name = model_name_default

    return dataset_dir_name, checkpoint_dir_name, stats_dir_name, model_name

def getData(data, training_testing_ratio):
    trainingShape = int(data.shape[0]*(training_testing_ratio/100 *1))
    training = data.iloc[:trainingShape]
    testing = data.iloc[trainingShape:]
    testing = testing.loc[:, ~testing.columns.str.contains('^Unnamed')]
    return training, testing

def calibrate(dataset_dir_name):
    data = pd.read_csv(f"{dataset_dir_name}/data.csv", sep ='|')
    ImgName = []
    for v in data['Image Name']:
        v = "D:/selfDrivingCar/dataset/ProcessedData/IMGs/" + v
        ImgName.append(v)
    data['Image Name'] = ImgName
    return data

def getMatrix(data):
    ImagesMatrix = []
    for v in data['Image Name']:
        img = imageio.imread(v, as_gray=False, pilmode="RGB")
        ImagesMatrix.append(img)

    return  np.array(ImagesMatrix)
