import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils import data


class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(1, width), nn.Tanh()]

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, t):  # t allows a single value (tensor 1,1)
        return self.linear_stack(t)


def boundary(model, tbc):
    # y(tbc) = 1
    # dydt(tbc) = 0

    ybc = model(tbc)

    dydt = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph=True)[0]

    return ybc - 1, dydt - 0 # we're returning these in residual form so that we don't need to compute it later


def datapoin


def residual(model, t, params):  # t: ncol x 1
    m, mu, k = params

    # evaluate the model
    y = model(t)

    # calculate the derivatives TODO: Why do we need the derivatives?
    dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    d2dydt2 = torch.autograd.grad(dydt, t, grad_outputs=torch.ones_like(dydt), create_graph=True)[0]

    residual_y = m * d2dydt2 + mu * dydt + k * y + k * y

    return residual_y


if __name__ == "__main__":
    m = 1
    mu = 0.1
    k = 1
    params = (m, mu, k)

    model = PINN(5, 16)
