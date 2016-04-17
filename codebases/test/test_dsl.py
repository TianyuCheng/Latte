#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.abspath(".."))

from latte.dsl import *

# neuron subtype
class WeightedNeuron(Neuron):

    """WeightedNeuron"""

    def __init__(self):
        super(WeightedNeuron, self).__init__()
        self.weights          = None     # type Vector
        self.gradient_weights = None     # type Vector
        self.bias             = None     # type Vector
        self.gradient_bias    = None     # type Vector

    def forward(self):
        # perform dot product of weights and inputs
        for i in (0, inputs[0].length, 1):
            value += weights[i] * inputs[0][i]
        # add the bias
        value += bias[0]

    def backward(self):
        # compute back propagated gradient
        for i in (0, inputs[0].length, 1):
            gradient_inputs[0][i] += weights[i] * gradient
        # compute weight gradient
        for i in (0, inputs[0].length, 1):
            gradient_weights[i] += weights[i] * gradient
        # compute the bias gradient
        gradient_bias[0] += gradient


# scripts to generate DNN

# create network
net = Network()

# create ensembles of neurons
ensemble1 = net.create_ensemble(WeightedNeuron, 10)
ensemble2 = net.create_ensemble(WeightedNeuron, 10)
ensemble3 = net.create_ensemble(WeightedNeuron, 10)
ensemble4 = net.create_ensemble(WeightedNeuron, 10)

# add connections between ensembles
net.add_connections(ensemble1, ensemble2, [(i, j) for i in range(10) for j in range(10)])
net.add_connections(ensemble2, ensemble3, [(i, j) for i in range(10) for j in range(10)])

net.solve()
