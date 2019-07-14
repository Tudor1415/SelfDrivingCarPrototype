import tensorflow
import glob
import collections
import os
import pandas as pd
import pygal
from collections import Counter
import shutil
import helpers

dir_name_default = "D:/dataset/IMGs"
dest_dir_name_default = "D:/dataset/ProcessedData"

SteeringAngles = []
FrameNum = []
ImageName = []
Files = []
Throttle = []


dir_name = input("Please input your images dir:")
dest_dir_name = input("Please input your destination dir:")
statistics = input("Do you want statistics ?")

if not dir_name:
    dir_name = dir_name_default

if not dest_dir_name:
    dest_dir_name = dest_dir_name_default

for file in os.listdir(dir_name):
    if file.endswith(".jpg"):
        cut = file.split('_')
        SteeringAngles.append(cut[5])
        FrameNum.append(int(cut[1]))
        Throttle.append(cut[3])
        Files.append(file)

helpers.moveFiles(dest_dir_name, dir_name, Files, ImageName)

Data = pd.DataFrame({
                        "ImageName":ImageName,
                        "Frame":FrameNum,
                        "SteeringAngles":SteeringAngles,
                        "Throttle": Throttle,
                        })

helpers.writeCsv(dest_dir_name, Data)

'''
Statistics Section
'''
if statistics == 'y' or statistics == 'yes':
    ChartData = [round(float(x), 1) for x in SteeringAngles]
    MostCommon = Counter(ChartData).most_common(10)

    SteeringAnglesChart = pygal.Bar()
    SteeringAnglesChart.title = "Steering Angles"

    for i in MostCommon:
        SteeringAnglesChart.add(str(i[0]), i[1])

    SteeringAnglesChart.render_to_file('../charts/SteeringAnglesChart.svg')

print("---------------------------------")
print("Data preparation was succesfull !")
print("---------------------------------")