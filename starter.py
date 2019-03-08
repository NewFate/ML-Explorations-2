import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    x[x<0] = 0
    return x

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def computeLayer(X, W, b):
    return np.matmul(W, X) + b

def CE(target, prediction):
    newP = softmax(prediction)
    m = target.shape[0]
    log_likelihood = -np.log(newP[range(m),target])
    loss = np.sum(log_likelihood) / m
    return loss
    

def gradCE(target, prediction):
    np.apply_along_axis(softmax, 0, prediction)
    return (-1)*np.sum(target/prediction)

def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();

    trainData = np.reshape(trainData,(10000, 784))  
    validData = np.reshape(validData, (6000, 784))    
    testData = np.reshape(testData, (2724, 784))
    
    print("Train Data shape:", trainData.shape)
    print("Train Target shape:", trainTarget.shape)
    print("Valid Data shape:", validData.shape)
    print("Valid Target shape:", validTarget.shape)
    print("Test Data shape:", testData.shape)
    print("Test Target shape:", testTarget.shape)
    
    #One hot encoding
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    print(trainTarget)
    print("Train Target encoded shape: ", trainTarget.shape)
    print("Valid Target encoded shape: ", validTarget.shape)
    print("Test Target encoded shape: ", testTarget.shape)  

    
    
if __name__ == "__main__":
    main()