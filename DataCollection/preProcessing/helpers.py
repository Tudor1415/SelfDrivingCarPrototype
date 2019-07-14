import os
import pandas as pd
def getLastImg(dest_dir_name):
    indexes = []
    allImages = os.listdir(f"{dest_dir_name}/IMGs")
    for i in allImages:
        j = i.split('_')[1]
        j = int(j.split('.')[0])
        indexes.append(j)

    return max(indexes) + 1


def moveFiles(dest_dir_name, dir_name, Files, ImageName):
    if len(os.listdir(f"{dest_dir_name}/IMGs") ) == 0:
        i = 0
        for f in Files:
            os.rename(f"{dir_name}/{f}", f"{dest_dir_name}/IMGs/Image_{i}.jpg")
            ImageName.append('Image_' + str(i) + '.jpg')
            i+=1
    else:
        i = getLastImg(dest_dir_name)
        for f in Files:
            os.rename(f"{dir_name}/{f}", f"{dest_dir_name}/IMGs/Image_{i}.jpg")
            ImageName.append(f"{dest_dir_name}/IMGs/Image_{i}.jpg")
            i+=1

def writeCsv(dest_dir_name, Data):
    if os.path.exists(f"{dest_dir_name}/data.csv") == False or os.stat(f"{dest_dir_name}/data.csv").st_size <= 1:
        Data.to_csv(f"{dest_dir_name}/data.csv", sep='|', encoding='utf-8')
    else:
        DataExistent = pd.read_csv(f"{dest_dir_name}/data.csv", sep='|')
        frames = [DataExistent, Data]
        result = pd.concat(frames, sort = False)
        result = result.loc[:, ~result.columns.str.contains('^Unnamed')]
        result.drop(result.columns[1], axis=1)
        os.remove(f"{dest_dir_name}/data.csv")
        result.to_csv(f"{dest_dir_name}/data.csv", sep='|', encoding='utf-8')