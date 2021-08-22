#!/usr/bin/env python3 

#
# TestDataset -- RDP much of this code was provided but I modified it to add OR and AND testcases and a test routine. 
#  
#

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


class TestDataset(Dataset):
    
    # Constructor
    # N_s is the size of the dataset
    #
    # This creates x, which is a N_s by 2 array pf points
    #  1/4 of those points will be in each corner. [(0,0),(0,1),(1,0),(1,1)] + a random amount of noise. 
    #
    def __init__(self, gateType, N_s=100):
        self.gateType = gateType
        # Create a N_s by 2 array for the X values representing the coordinates
        self.x = torch.zeros((N_s, 2))
        # Create a N_s by 1 array for the class the X value belongs to
        self.y = torch.zeros((N_s, 1))
        # Split the dataset into 4 sections
        if gateType == "OR":
          OneZeroCase = 1.0 
          ZeroOneCase = 1.0 
          OneOneCase = 1.0 
        elif gateType == "AND":
          OneZeroCase = 0.0 
          ZeroOneCase = 0.0 
          OneOneCase = 1.0 
        elif gateType == "XOR":
          OneZeroCase = 1.0 
          ZeroOneCase = 1.0 
          OneOneCase = 0.0 
        for i in range(N_s // 4):
            # Create data centered around (0,0) of class 0
            self.x[i, :] = torch.Tensor([0.0, 0.0]) 
            self.y[i, 0] = torch.Tensor([0.0])

            # Create data centered around (0,1) of class 1
            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([ZeroOneCase])
    
            # Create data centered around (1,0) of class 1
            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([OneZeroCase])
    
            # Create data centered around (1,1) of class 0
            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([OneOneCase])

            # Add some noise to the X values to make them different
            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s

    def name(self):
       return self.gateType

    # Getter
    def __getitem__(self, index):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
    # Plot the data
    def plot_stuff(self):
        plt.plot(self.x[self.y[:, 0] == 0, 0].numpy(), self.x[self.y[:, 0] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:, 0] == 1, 0].numpy(), self.x[self.y[:, 0] == 1, 1].numpy(), 'ro', label="y=1")
        plt.legend()

    def testSet(self,model):
      testInputs = [[0,0],[1,0],[0,1],[1,1]]
      if (self.gateType == 'XOR'): 
        expected   = [   0,    1,    1,   0 ]
      elif (self.gateType == 'OR'): 
        expected   = [   0,    1,    1,   1 ]
      elif (self.gateType == 'AND'): 
        expected   = [   0,    0,    0,   1 ]
      for inputNum in range(0,4):
        prediction = model(torch.Tensor(testInputs[inputNum])).detach().numpy()[0]
        expect     = expected[inputNum]
        if (abs(prediction - expect) > 0.2):
          print("Should be ",expect," was ",prediction)
          return False
      return True



