#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.abspath(".."))

from latte import *

# neuron subtype
class WeightedNeuron(Neuron):

    """WeightedNeuron"""

    def __init__(self):
        super(WeightedNeuron, self).__init__()

    def forward(self):
        print "This implements the forward algorithm for WeightedNeuron" 

    def backward(self):
        print "This implements the backward algorithm for WeightedNeuron" 


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
