import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Variable
figure_number = 0

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

def softmax(x):
    for i in range(x.shape[0]):
        x[i] = np.exp(x[i] - np.max(x[i]))
        e_x_sum = np.sum(x[i])
        x[i] = x[i] / e_x_sum
    return x # only difference

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
	return -1/(target.shape[0])*np.sum(1/np.dot(prediction, target))


def find_accuracy(target, prediction):
    correct_classification = 0
    target_length = target.shape[0]
    for i in range(target_length):
        max_predict = np.argmax(prediction[i], axis=0)
        max_label = np.argmax(target[i], axis=0)
        if(max_predict == max_label):
            correct_classification += 1
    return correct_classification/target_length


def gradientDescentMomentum(trainData, trainTarget, validData, validTarget, testData, testTarget, weight_hidden, weight_output, bias_hidden, bias_output, epochs, hidden_unit_size):
    #v matrices
    v_hidden = np.ones((784, hidden_unit_size))*1e-5
    v_output = np.ones((hidden_unit_size, 10))*1e-5
    
    lamda = 0.9
    alpha = 0.00001
    
    #Accuracy need to complete
    accuracy = 0
    
    #iterations
    i = 0
    
    #Loss values
    train_loss_list = []
    train_accuracy_list = []
    valid_loss_list = []
    valid_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    
    while i < epochs:
        
        #Run front propagation
        loss_train, accuracy_train, y_predict_train, hidden_layer_activation_train = frontPropagation(trainData, trainTarget, weight_hidden, weight_output, bias_hidden, bias_output)
        loss_valid, accuracy_valid, y_predict_valid, hidden_layer_activation_valid = frontPropagation(validData, validTarget, weight_hidden, weight_output, bias_hidden, bias_output)
        loss_test, accuracy_test, y_predict_test, hidden_layer_activation_test = frontPropagation(testData, testTarget, weight_hidden, weight_output, bias_hidden, bias_output)
        
        print("Iteration: ",i, "Train Loss: ",loss_train, "Validation Loss: ",loss_valid, "Test Loss: ",loss_test)
        print("Train Accuracy: ",accuracy_train, "Valid Accuracy: ",accuracy_valid, "Test Accuracy: ",accuracy_test)
        
        train_loss_list.append(loss_train)
        train_accuracy_list.append(accuracy_train)
        
        valid_loss_list.append(loss_valid)
        valid_accuracy_list.append(accuracy_valid)
        
        test_loss_list.append(loss_test)
        test_accuracy_list.append(accuracy_test)
    
        dL_dWo, dL_dWh = backPropagation(trainData, trainTarget, y_predict_train, hidden_layer_activation_train, weight_output)
        
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
    
    
    return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_loss_list, test_accuracy_list


def backPropagation(trainData, trainTarget, y_predict, hidden_layer_activation, weight_output):
    
    #Calculate the partial derivatives
    delta_1 = np.transpose(y_predict-trainTarget) #10x10000
    #print(delta_1)
    dL_dWo = np.matmul(hidden_layer_activation.transpose(), delta_1.transpose())
    #dL_dWo = np.transpose(dL_dWo) #1000x10
    #print("dL_dWo: ", dL_dWo.shape)
    
    delta_2 = np.multiply(np.matmul(weight_output, delta_1).transpose(), np.sign(hidden_layer_activation)) #10000x1000
    #print(delta_2.shape)
    #delta_2 = np.transpose(delta_2)
    #print(delta_2)
    dL_dWh = np.matmul(trainData.transpose(), delta_2) #784x1000
    
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
    accuracy = find_accuracy(trainTarget, y_predict)
    
    return loss, accuracy, y_predict, hidden_layer_activation



def plot_loss_different_hidden_unit_size(train_loss_list, valid_loss_list, test_loss_list, figure_number, hidden_unit_size):
    # Plot the loss graphs for train, validation and test sets
    plt.figure(figure_number)
    plt.plot(train_loss_list, c='b', label=hidden_unit_size)
    plt.plot(valid_loss_list, c='r', label=hidden_unit_size)
    plt.plot(test_loss_list, c='g', label=hidden_unit_size)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('LOSS (Varying Hidden Unit Size)')
    
    
def plot_accuracy_different_hidden_unit_size(train_accuracy_list, valid_accuracy_list, test_accuracy_list, figure_number, hidden_unit_size):
    # Plot the loss graphs for train, validation and test sets
    plt.figure(figure_number)
    plt.plot(train_accuracy_list, c='b', label=hidden_unit_size)
    plt.plot(valid_accuracy_list, c='r', label=hidden_unit_size)
    plt.plot(test_accuracy_list, c='g', label=hidden_unit_size)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (Varying Hidden Unit Size)')


    ########End of Neural Network#####
    
    
    ########Convolutional Neural Network Functions######


def create_convolution_layer(input, num_input_channels, filter_size, num_filters, name):    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        
        #weights = tf.Variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights


def create_pool_layer(input, name):    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')        
        return layer
    
    
    
def create_relu_layer(input, name):    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer


def new_fc_layer(input, num_inputs, num_outputs, name):    
    with tf.variable_scope(name) as scope:

        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases 
        
        return layer


def create_batch_normalisation_layer(input, name):    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for batch normalisation
        input_mean, input_var = tf.nn.moments(input,[0])
        scale = tf.Variable(tf.ones([32]))
        beta = tf.Variable(tf.zeros([32]))
        BN = tf.nn.batch_normalization(input, input_mean, input_var, beta, scale, 1e-3)
        
        return BN


def create_flatten_layer(input, name):    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for flatten
        layer = tf.layers.flatten(input)
        
        return layer

def create_batch(data, batch_size):
    while data.any():
        batch, data = data[:batch_size], data[batch_size:]
        yield batch

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
    
    hidden_unit = [1000, 100, 500, 2000]
    #Hidden layer
    
    #Hidden layer weight initialisation
    figure_number = 1
    
    for hidden_unit_size in hidden_unit:
        units_in = 784
        units_out = hidden_unit_size
        w_hidden = np.random.randn(784, hidden_unit_size)*np.sqrt(2/(units_in+units_out))
        #w_hidden = w_hidden.reshape(784, 1000)
        #print(w_hidden)
        
        #Hidden layer bias iniatilisation
        b_hidden = np.ones((1, hidden_unit_size))
        
        #Ouput layer
         
        #Output layer weight initialization
        units_in = hidden_unit_size
        units_out = 10
        w_output = np.random.randn(hidden_unit_size, 10)*np.sqrt(2/(units_in+units_out))
        #w_output = w_output.reshape(1000, 10)
        #print(w_output)
        
        #Output layer bias initialization
        b_output = np.ones((1, 10))
        #print(b_output)
        
        #Gradient Descent with momentum
        train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, test_loss_list, test_accuracy_list = gradientDescentMomentum(trainData, trainTarget, validData, validTarget, testData, testTarget, w_hidden, w_output, b_hidden, b_output, 0, hidden_unit_size)
        
        plot_loss_different_hidden_unit_size(train_loss_list, valid_loss_list, test_loss_list, figure_number, hidden_unit_size)
        figure_number +=1
        
        plot_accuracy_different_hidden_unit_size(train_accuracy_list, valid_accuracy_list, test_accuracy_list, figure_number, hidden_unit_size)
        figure_number +=1
        
        print("Finished")

    
    
    
    #########################################################
    #                                                       #
    #           Convolutional Neural Network                #
    #                                                       #
    #                                                       #
    #########################################################
    
    # Placeholder variable for the input images
    x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
    # Reshape it into [num_images, img_height, img_width, num_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Placeholder variable for the true labels associated with the images
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    # Convolutional Layer 1
    layer_conv1, weights_conv1 = create_convolution_layer(input=x_image, num_input_channels=1, filter_size=3, num_filters=32, name ="conv1")
    
    # RelU layer 1
    layer_relu1 = create_relu_layer(layer_conv1, name="relu1")
    
    # Batch Normalisation Layer
    layer_BN = create_batch_normalisation_layer(layer_relu1, name="BN")
       
    # Pooling Layer 1
    layer_pool1 = create_pool_layer(layer_BN, name="pool")
    
    # Flatten layer
    num_features = layer_pool1.get_shape()[1:4].num_elements()
    layer_flatten = tf.reshape(layer_pool1, [-1, num_features])
    
    # Fully-Connected Layer 1
    layer_fc1 = new_fc_layer(layer_flatten, num_inputs=num_features, num_outputs=784, name="fc1")
    
    # RelU layer 2
    layer_relu2 = create_relu_layer(layer_fc1, name="relu2")
    
    # Fully-Connected Layer 2
    layer_fc2 = new_fc_layer(layer_relu2, num_inputs=784, num_outputs=10, name="fc2")
      
    
    # Use Softmax function to normalize the output
    with tf.variable_scope("softmax"):
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        
    
    # Use Cross entropy cost function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        

    # Use Adam Optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
               
    # Accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    num_epochs = 5
    batch_size = 32
    train_accuracy_list = []
    train_loss_list = []
    
    valid_accuracy_list = []
    valid_loss_list = []
    
    test_accuracy_list = []
    test_loss_list = []
    
    batch_train_accuracy = []
    batch_train_loss = []
            
    batch_valid_loss = []
    batch_valid_accuracy = []
            
    batch_test_loss = []
    batch_test_accuracy = []
    
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Loop over number of epochs
        for epoch in range(num_epochs):

            
            
            # Shuffle trainData and trainTarget
            trainData, trainTarget = shuffle(trainData, trainTarget)
            for (batch_x, batch_y) in zip(create_batch(trainData, batch_size), create_batch(trainTarget, batch_size)):
            
                #start_time = time.time()
        
                
                # Put the batch into a dict with the proper names for placeholder variables
                feed_dict_train = {x: batch_x, y_true: batch_y}
                
                # Run the optimizer using this batch of training data.
                sess.run(optimizer, feed_dict=feed_dict_train)
                
                # Calculate the loss and accuracy on the batch of training data
                #batch_train_accuracy.append(sess.run(accuracy, feed_dict=feed_dict_train))
                train_loss, train_accuracy = sess.run([cost, accuracy], feed_dict=feed_dict_train)
                #valid_loss, valid_accuracy = sess.run([cost, accuracy], feed_dict={x: validData, y_true: validTarget})
                #test_loss, test_accuracy = sess.run([cost, accuracy], feed_dict={x: testData, y_true: testTarget})
                
                batch_train_loss.append(train_loss)
                batch_train_accuracy.append(train_accuracy)
                
                #batch_valid_loss.append(valid_loss)
                #batch_valid_accuracy.append(valid_accuracy)
                
                #batch_test_loss.append(test_loss)
                #batch_test_accuracy.append(test_accuracy)
                
            
              
            train_accuracy_mean = np.mean(batch_train_accuracy)
            train_loss_mean = np.mean(batch_train_loss)
            
            train_accuracy_list.append(train_accuracy_mean)
            train_loss_list.append(train_loss_mean)
            
            #valid_accuracy_mean = np.mean(batch_valid_accuracy)
            #valid_loss_mean = np.mean(batch_valid_loss)
            
            #valid_accuracy_list.append(valid_accuracy_mean)
            #valid_loss_list.append(valid_loss_mean)
            
            #test_accuracy_mean = np.mean(batch_test_accuracy)
            #test_loss_mean = np.mean(batch_test_loss)
            
            #test_accuracy_list.append(test_accuracy_mean)
            #test_loss_list.append(test_loss_mean)
        
            
            print("Train Accuracy: ", train_accuracy_mean, "Train Loss: ", train_loss_mean)
            #print("Valid Accuracy: ", train_accuracy_mean, "Valid Loss: ", train_loss_mean)
            #print("Test Accuracy: ", train_accuracy_mean, "Test Loss: ", train_loss_mean)

        valid_loss, valid_accuracy = sess.run([cost, accuracy], feed_dict={x: validData, y_true: validTarget})
        test_loss, test_accuracy = sess.run([cost, accuracy], feed_dict={x: testData, y_true: testTarget})

    
if __name__ == "__main__":
    main()