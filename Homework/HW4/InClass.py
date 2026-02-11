import torch
import numpy as np
from torchdiffeq import odeint
from torch import nn
import matplotlib.pyplot as plt


class ODENet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ODENet, self).__init__()
        inpu


if __name__ == '__main__':
    data = np.loadtxt("InClassData.txt")
    t_train = torch.tensor(data[:, 0]) #nt
    y_train = torch.tensor(data[:, 1:]) #nt x 2
    plt.plot(t_train, y_train[:,0])
    plt.plot(t_train, y_train[:,1])
    plt.show()



