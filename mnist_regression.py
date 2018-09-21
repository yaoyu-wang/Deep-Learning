import numpy as np
import math as math
import tensorflow as tf
import matplotlib.pyplot as plt
% matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Dataset statistics# Datas 
print('Training image data: ', mnist.train.images.shape)
print('Testing image data: ', mnist.test.images.shape)
print('28 x 28 = ', 28*28)

# Example image
print('\nTrain image 1 is labelled one-hot as {0}'.format(mnist.train.labels[1,:]))
image = np.reshape(mnist.train.images[1,:],[28,28])
#plt.imshow(image, cmap='gray')

current_data=mnist.train.next_batch(1)
# Example image
print('\nTrain image 1 is labelled one-hot as {0}'.format(current_data[1]))
image = np.reshape(current_data[0],[28,28])
#plt.imshow(image, cmap='gray')

def lr_gradient(W,b,data,target):
    # calculate the gradient on the data
    gamma=np.dot(W,data.T)+b
    s_max=softmax(gamma,target)
    dL_dgamma=s_max
    for i in range(0,10):
        dL_dgamma[i]=dL_dgamma[i]-target[i]
    W_grad=np.dot(dL_dgamma,data)
    b_grad=np.mean(dL_dgamma)
    return W_grad, b_grad

def lr_loss(W,b,data,target):
    # calculate the loss
    gamma=np.dot(W,data.T)+b
    sum_of_exp=sum_exp(gamma)
    single_loss=np.random.rand(10,1)
    for i in range(0,10):
        single_loss[i]=math.log(math.exp(gamma[i])/sum_of_exp)
    avg_loss=-(1.0/target.shape[0])*np.sum(single_loss)
    return avg_loss

def softmax(gamma,target):
    sum_of_exp=sum_exp(gamma)
    dL_dgamma=np.random.rand(10,1)
    for i in range(0,10):
        dL_dgamma[i]=math.exp(gamma[i])/sum_of_exp
    return dL_dgamma

def sum_exp(gamma):
    result=0
    for i in range(0,10):
        result+=math.exp(gamma[i])
    return result

def acc_and_loss(W,b,data,target):
    result=np.array(softmax(np.dot(W,data.T)+b,target))
    loss=lr_loss(W,b,data,target)
    
    index=0
    max=result[0]
    for i in range(1,10):
        if result[i]>max:
            max=result[i]
            index=i
    
    if target[index]==1:
        acc=1
    else:
        acc=0

    return acc,loss

max_iterations=200000 # choose the max number of iterations
step_size=0.02 # choose your step size
W=np.random.rand(10,784)/100  # choose your starting parameters (connection weights)
b=np.random.rand(10,1)/100  # choose your starting parameters (biases)
num_of_accuracy=0
training_loss_history=[]
for iter in range(0,max_iterations):
    data,target=mnist.train.next_batch(1)
    # note you need to change this to your preferred data format.
    W_grad,b_grad=lr_gradient(W,b,data,target[0])
    loss=lr_loss(W,b,data,target[0])
    if iter%100==0:
        training_loss_history.append(loss)
    W=W-step_size*W_grad
    b=b-step_size*b_grad
    if iter%10000==0:
        step_size*=0.9
    num_of_accuracy+=acc_and_loss(W,b,data,target[0])[0]
    
plt.plot(training_loss_history)
plt.pause(5)
    
current_data=mnist.test.next_batch(1)
print('\nTrain image 1 is labelled one-hot as {0}'.format(current_data[1]))
gamma=np.dot(W,np.linalg.pinv(current_data[0]))+b;
#print("gamma: {0}".format(gamma))
image = np.reshape(current_data[0],[28,28])
plt.imshow(image, cmap='gray')

# Calculate both your training loss and accuracy and your validation loss and accuracy 
# Fill in code here.
#training (Process above)
print("training accuracy: ",num_of_accuracy/max_iterations)
print("training loss: ",np.mean(training_loss_history))

#validation
validation_loss_record=[]
accuracy_validation=0
for iter in range(0,5000):
    data,target=mnist.train.next_batch(1)
    result=acc_and_loss(W,b,data,target[0])
    accuracy_validation+=result[0]
    validation_loss_record.append(result[1])
print("validation accuracy: ",accuracy_validation/5000)
print("validation loss: ",np.mean(validation_loss_record))
