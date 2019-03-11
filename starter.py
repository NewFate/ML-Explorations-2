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
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    print(softmax.shape)
    s = softmax.reshape(-1,1)
    print(s.shape)
    return np.diagflat(s) - np.dot(s, s.T)

def computeLayer(X, W, b):
    return np.matmul(X, W) + b


def CE(target, prediction):
    #prediction = softmax(prediction)
    log_likelihood = np.log(prediction)
    loss = -np.sum(log_likelihood*target) / target.shape[0]
    return loss
 
    
def gradCE(target, prediction):
	#print("Here0")
	return -1/(target.size)*np.sum(1/np.dot(prediction*target))


def gradientDescentMomentum(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output, epochs):
    #v matrices
    v_hidden = np.ones((784, 1000))*1e-5
    v_output = np.ones((1000, 10))*1e-5
    
    lamda = 0.9
    alpha = 0.0001
    
    #Accuracy need to complete
    accuracy = 0
    
    #iterations
    i = 0
    
    #Loss values
    loss_list = []
    
    while i < epochs:
        
        #Run front propagation
        loss, y_predict, hidden_layer_activation = frontPropagation(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output)
        
        print("Iteration: ",i, "Loss: ",loss)
        
        loss_list.append(loss)
    
        dL_dWo, dL_dWh = backPropagation(trainData, trainTarget, y_predict, hidden_layer_activation, weight_output)
        
        #Momentum based updates
        
        #Update hidden layer weight
        v_hidden = lamda*v_hidden + alpha*dL_dWh
        #print(v_hidden)
        weight_hidden = weight_hidden - v_hidden
        #print("Hidden weight: ", weight_hidden)
        
        #Update output layer weight
        v_output = lamda*v_output + alpha*dL_dWo
        #print(v_output)
        weight_output = weight_output - v_output
        #print("Output weight: ", weight_output)

        i += 1
    
    plt.plot(loss_list)
    
    return y_predict, loss_list, accuracy


def backPropagation(trainData, trainTarget, y_predict, hidden_layer_activation, weight_output):
    
    #Calculate the partial derivatives
    delta_1 = np.transpose(y_predict-trainTarget)
    #print(delta_1)
    dL_dWo = np.matmul(hidden_layer_activation.transpose(), delta_1.transpose())
    #dL_dWo = np.transpose(dL_dWo) #1000x10
    #print("dL_dWo: ", dL_dWo.shape)
    
    delta_2 = np.multiply(np.matmul(weight_output, delta_1).transpose(), np.sign(hidden_layer_activation)) #10000x1000
    #delta_2 = np.transpose(delta_2)
    #print(delta_2)
    dL_dWh = np.matmul(trainData.transpose(), delta_2) #1000x784
    #dL_dWh = np.transpose(dL_dWh)
    #print("dL_dWh: ", dL_dWh.shape)

    
    return dL_dWo, dL_dWh
    


def frontPropagation(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output):
    #Hidden layer computation
    hidden_layer = computeLayer(trainData, weight_hidden, bias_hidden)
    #print("Hidden layer: ",hidden_layer)
    hidden_layer_activation = relu(hidden_layer) #10000x1000
    #print("Hidden layer activation: ", hidden_layer_activation)
    
    #Ouput layer computation
    output_layer = computeLayer(hidden_layer_activation, weight_output, bias_output)
    #print("Output layer: ",output_layer)
    #print(output_layer)
    output_layer_activation = softmax(output_layer)
    #print("Output layer activation: ", output_layer_activation)
    
    y_predict = output_layer_activation
    #print(y_predict)
    loss = CE(trainTarget, y_predict)
    
    return loss, y_predict, hidden_layer_activation
    

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
    units_in = 784
    units_out = 1000
    w_hidden = np.random.randn(784, 1000)*np.sqrt(2/(units_in+units_out))
    #w_hidden = w_hidden.reshape(784, 1000)
    #print(w_hidden)
    
    #Hidden layer bias iniatilisation
    b_hidden = np.ones((1, 1000))
    
    #Ouput layer
     
    #Output layer weight initialization
    units_in = 1000
    units_out = 10
    w_output = np.random.randn(1000, 10)*np.sqrt(2/(units_in+units_out))
    #w_output = w_output.reshape(1000, 10)
    #print(w_output)
    
    #Output layer bias initialization
    b_output = np.ones((1, 10))
    #print(b_output)
    
    #Gradient Descent with momentum
    y_predict, loss_list, accuracy = gradientDescentMomentum(trainData, trainTarget, w_hidden, w_output, b_hidden, b_output, 200)
    print("Finished")

    
    
if __name__ == "__main__":
    main()