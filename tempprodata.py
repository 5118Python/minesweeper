import argparse
import os

import numpy as np
import tensorflow as tf

from gendata import GenData

trainingData = np.load('F:\\minesweeper\\training2.npz')

trainingAroundData = trainingData["aroundData"]
trainingPredictData = trainingData["predictData"]

newAroundData1=[]
newPredictData1=[]

newAroundData2=[]
newPredictData2=[]

notPredictCount = 1
for index in range(len(trainingPredictData)):
    tempPredictData = trainingPredictData[index]
    tempAroundData = trainingAroundData[index]

    if tempPredictData[0] == 1:
        # 可以预测
        newAroundData1.append(tempAroundData)
        newPredictData1.append(tempPredictData)

    else:
        # 无法预测更少
        newAroundData2.append(tempAroundData)
        newPredictData2.append(tempPredictData)

        notPredictCount = notPredictCount + 1

finalAroundData=[]
finalPredictData=[]

newAroundData2Len = len(newAroundData2)
finalAroundLen = len(newAroundData1) * 2

while len(newAroundData1) < finalAroundLen:
    subLen = finalAroundLen - len(newAroundData1)
    appendCount = min(subLen, newAroundData2Len)
    appendList = newAroundData2[0:appendCount]
    appendPredictList = newPredictData2[0:appendCount]

    newAroundData1 = newAroundData1 + appendList
    newPredictData1 = newPredictData1 + appendPredictList

newAroundData1 = np.array(newAroundData1)
newPredictData1 = np.array(newPredictData1)

np.savez('F:\\minesweeper\\new_training2.npz', aroundData=newAroundData1, predictData=newPredictData1)

print(len(newAroundData1))

