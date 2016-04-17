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
        for i in xrange(len(self.inputs[0])):
            self.value += self.weights[i] * self.inputs[0][i]
        # add the bias
        self.value += self.bias[1]

    def backward(self):
        # compute back propagated gradient
        for i in xrange(len(self.inputs[0])):
            self.gradient_inputs[0][i] += self.weights[i] * self.gradient
        # compute weight gradient
        for i in xrange(len(self.inputs[0])):
            self.gradient_weights[i] += self.weights[i] * self.gradient
        # compute the bias gradient
        self.gradient_bias += self.gradient


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
