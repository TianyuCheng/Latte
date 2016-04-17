#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.join(os.getcwd(), "../"))

from latte import *

class WeightedNeuron(Neuron):

    """WeightedNeuron"""

    def __init__(self):
        super(WeightedNeuron, self).__init__()

    def forward(self):
       print "This implements the forward algorithm for WeightedNeuron" 

    def backward(self):
       print "This implements the backward algorithm for WeightedNeuron" 
