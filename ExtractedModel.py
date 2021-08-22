#!/usr/bin/python3

#
# Here we pull the weights and biases from the neural networks we just built # and multiply them out ...
#
#
from collections import OrderedDict
from torch import tensor 
import numpy as np 
import torch 

#
# We generated these when we ran ./SimpleNeuralNetwork.py 
#

oneNeuronModelText = "OrderedDict([('linear1.weight', tensor([[-8.7824, -9.1485]])), ('linear1.bias', tensor([1.3729])), ('linear2.weight', tensor([[-8.0385]])), ('linear2.bias', tensor([-0.1427]))])"

twoNeuronModelText = "OrderedDict([('linear1.weight', tensor([[ 6.8541, -7.3292], [-7.4768,  7.3833]])), ('linear1.bias', tensor([-3.6507, -4.0413])), ('linear2.weight', tensor([[13.4552, 13.4148]])), ('linear2.bias', tensor([-6.6226]))])"


oneNeuronInfo = eval(twoNeuronModelText)
twoNeuronInfo = eval(twoNeuronModelText)

# Get the weights and biases for one-neuron case 
N1wT1 = oneNeuronInfo['linear1.weight'].numpy()[0]
N1b1  = oneNeuronInfo['linear1.bias'].numpy()[0]
N1wT2 = oneNeuronInfo['linear2.weight'].numpy()[0]
N1b2  = oneNeuronInfo['linear2.bias'].numpy()[0]

# Get the weights and biases for two-neuron case 
N2wT1a = twoNeuronInfo['linear1.weight'].numpy()[0]
N2b1a  = twoNeuronInfo['linear1.bias'].numpy()[0]
N2wT1b = twoNeuronInfo['linear1.weight'].numpy()[1]
N2b1b  = twoNeuronInfo['linear1.bias'].numpy()[1]
N2wT2  = twoNeuronInfo['linear2.weight'].numpy()[0]
N2b2   = twoNeuronInfo['linear2.bias'].numpy()[0]

def oneNeuronModel(X):
   A   = torch.sigmoid(torch.Tensor([X[0]*N1wT1[0] + X[1]*N1wT1[1]+N1b1]))
   return torch.sigmoid(torch.Tensor(A*N1wT2+N1b2)).numpy()[0]

def twoNeuronModel(X):
   twoNeuronInfo = eval(oneNeuronModelText)
   # For an indefinite length of vector I would of course use numpy arrays or some other vector multiplication
   A   = torch.sigmoid(torch.Tensor([X[0]*N2wT1a[0] + X[1]*N2wT1a[1]+N2b1a]))
   B   = torch.sigmoid(torch.Tensor([X[0]*N2wT1b[0] + X[1]*N2wT1b[1]+N2b1b]))
   return torch.sigmoid(torch.Tensor(A*N2wT2[0]+B*N2wT2[1]+N2b2)).numpy()[0]



print("Results for XOR of two numbers in a 1 and 2 hidden neuron case")
print(" I generated the weights and biases from SimpleNeuralNetwork.py ") 

for input in [[0,0],[1,0],[0,1],[1,1]]:
  print("XOR ",input," ", oneNeuronModel(input))
  print("XOR ",input," ", twoNeuronModel(input))
