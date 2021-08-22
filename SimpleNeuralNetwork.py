#!/usr/bin/env python3 

# Import the libraries we need for this lab

# Allows us to use arrays to manipulate and store data
import numpy as np
# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to use activation functions
import torch.nn.functional as F
# Used to graph data and loss curves
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
# Used to help create the dataset and perform mini-batch
from torch.utils.data import Dataset, DataLoader

#
# Plots the decision boundaries. 
#

def plot_decision_regions_2class(model,data_set,filename):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1 , X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1 , X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, shading='auto')
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()
    plt.savefig(filename)

#
# Accuracy error function. 
#

def accuracy(model, data_set):
    # Rounds prediction to nearest integer 0 or 1
    # Checks if prediction matches the actual values and returns accuracy rate
    return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())


#
# Define the class Net with H nodes in the hidden layer
# D_in is the number of input parameters. 
# D_out is the number of input parameters. 
#

class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # D_in is the input size of the first layer (size of input layer)
        # H is the outpout size of the first layer and the input size of the second layer (size of hidden layer)
        # D_out is the output size of the second layer (size of output layer)
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction    
    def forward(self, x):
        # Puts x through first layer then sigmoid function
        x = torch.sigmoid(self.linear1(x)) 
        # Puts result of previous line through second layer then sigmoid function
        x = torch.sigmoid(self.linear2(x))
        # Output is a number between 0 and 1 due to the sigmoid function. Whichever the output is closer to, 0 or 1, is the class prediction
        return x


#
#
# Train the model. 
#
#

def train(data_set, model, criterion, train_loader, optimizer, epochs=5):
    # Lists to keep track of cost and accuracy
    COST = []
    ACC = []
    # Number of times we train on the entire dataset
    for epoch in range(epochs):
        # Total loss over epoch
        total=0
        # For batch in train laoder
        for x, y in train_loader:
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction based on X value
            yhat = model(x)
            # Measures the loss between prediction and acutal Y value
            loss = criterion(yhat, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Cumulates loss 
            total+=loss.item()
        # Saves cost and accuracy
        ACC.append(accuracy(model, data_set))
        COST.append(total)
        
    # Prints Cost vs Epoch graph
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)
    
    # Prints Accuracy vs Epoch graph
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
#    plt.show()

    return COST

from TestDataset import TestDataset 


for dataSetType in ["XOR","AND","OR"]:
  if (dataSetType == "XOR"):
    neurons = [1,2]
  else: 
    neurons = [1]
  for hiddenNeurons in neurons:
    data_set = TestDataset(dataSetType)
    model    = Net(2,hiddenNeurons,1)
    learning_rate = 0.1

    # We create a criterion which will measure loss
    criterion = nn.BCELoss()
    # Create an optimizer that updates model parameters using the learning rate and gradient
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
    # Create a Data Loader for the training data with a batch size of 1 
    train_loader = DataLoader(dataset=data_set, batch_size=1)

    # Using the training function train the model on 500 epochs
    LOSS12 = train(data_set, model, criterion, train_loader, optimizer, epochs=500)
  
    result = data_set.testSet(model)
    if result: 
      resultText = "PASS"
    else: 
      resultText = "FAIL"

    if (dataSetType == "XOR") and (hiddenNeurons == 1):  
      print (" XOR with 1 neuron should FAIL:  " + resultText) 
    elif (dataSetType == "XOR") and (hiddenNeurons == 2):  
      print (" XOR with 2 neurons should PASS: " + resultText) 
    else: 
      print (dataSetType+" should PASS:         " + resultText) 


    # Plot the data with decision boundaries
    plot_decision_regions_2class(model, data_set,data_set.name()+"neurons"+str(hiddenNeurons)+".png")

    l=model.state_dict()
    print(l)  

