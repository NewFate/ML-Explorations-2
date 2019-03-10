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

def relu_grad(x):
    x[x<0] = 0
    return x

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def computeLayer(X, W, b):
    return np.matmul(X, W) + b


def CE(target, prediction):
    log_likelihood = np.log(prediction)
    loss = -np.sum(log_likelihood*target) / target.shape[0]
    return loss
 
    
def gradCE(target, prediction):
    np.apply_along_axis(softmax, 0, prediction)
    return (-1)*np.sum(target/prediction)


def gradientDescentMomentum(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output, epochs):
    #v matrices
    v_hidden = np.ones((784, 1000))*1e-5
    v_output = np.ones((1000, 10))*1e-5
    
    lamda = 0.9
    alpha = 0.01
    
    #Accuracy need to complete
    accuracy = 0
    
    #iterations
    i = 0
    
    while i < epochs:
        
        #Run front propagation
        loss, y_predict, output_layer, hidden_layer = frontPropagation(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output)
        
        print("Loss: ",loss)
    
        dL_dWo, dL_dWh = backPropagation(trainTarget, y_predict, output_layer, hidden_layer, weight_output)
        
        #Momentum based updates
        
        #Update hidden layer weight
        v_hidden = lamda*v_hidden + alpha*dL_dWh
        weight_hidden = weight_hidden - v_hidden
        
        #Update output layer weight
        v_output = lamda*v_output + alpha*dL_dWo
        weight_output = weight_output - v_output

        i += 1
    
    return y_predict, loss, accuracy


def backPropagation(trainTarget, y_predict, output_layer, hidden_layer, weight_hidden):
    
    #Calculate the deltas
    delta_3 = gradCE(trainTarget, y_predict)*softmax_grad(y_predict)
    delta_2 = delta_3*weight_hidden*relu_grad(hidden_layer)
    
    #Calculate the partial derivatives
    dL_dWo = y_predict*delta_3
    dL_dWh = output_layer*delta_2
    
    return dL_dWo, dL_dWh
    


def frontPropagation(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output):
    #Hidden layer computation
    hidden_layer = computeLayer(trainData, weight_hidden, bias_hidden)
    print("Hidden layer shape: ",hidden_layer.shape)
    hidden_layer_activation = relu(hidden_layer)
    
    #Ouput layer computation
    output_layer = computeLayer(hidden_layer_activation, weight_output, bias_output)
    print("Output layer shape: ",output_layer.shape)
    output_layer_activation = softmax(output_layer)
    
    y_predict = output_layer_activation
    loss = CE(trainTarget, y_predict)
    
    return loss, y_predict, output_layer, hidden_layer
    

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
    
    print(trainTarget)
    
    #One hot encoding
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    #print(trainData)
    print("Train Target encoded shape: ", trainTarget.shape)
    print("Valid Target encoded shape: ", validTarget.shape)
    print("Test Target encoded shape: ", testTarget.shape)  
    
    #neural network
    
    #Hidden layer
    
    #Hidden layer weight initialisation
    units_in = 10000
    units_out = 1000
    w_hidden = np.random.rand((784000))*np.sqrt(2/(units_in+units_out))
    w_hidden = w_hidden.reshape(784, 1000)
    #print(w_hidden)
    
    #Hidden layer bias iniatilisation
    b_hidden = np.ones((1, 1000))
    
    #Ouput layer
     
    #Output layer weight initialization
    units_in = 1000
    units_out = 10000
    w_output = np.random.rand((10000))*np.sqrt(2/(units_in+units_out))
    w_output = w_output.reshape(1000, 10)
    #print(w_output)
    
    #Output layer bias initialization
    b_output = np.ones((1, 10))
    #print(b_output)
    
    #Gradient Descent with momentum
    y_predict, loss, accuracy = gradientDescentMomentum(trainData, trainTarget, w_hidden, w_output, b_hidden, b_output, 200)
    print("Finished")
    print(loss)
    
    
if __name__ == "__main__":
    main()