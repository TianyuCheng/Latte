#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.join(os.getcwd(), "../latte"))

from latte import *

@neuron
def WeightedNeuron():
    params = { "name": "WeightedNeuron" }

print WeightedNeuron()
