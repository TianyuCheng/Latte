'''
    Testcase: fc-relu.py
        1. FC layer with tanh activation
        2. FC layer with ReLU activation
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm = LibsvmDataLayer(net, \
                    '../datasets/iris-scale-train.libsvm',  \
                    '../datasets/iris-scale-test.libsvm', 1, 4, 3)
ip1_enm = FullyConnectedLayer(net, data_enm, 1, 10, FCNeuron)
ip2_enm = FullyConnectedLayer(net, ip1_enm, 1, 10, ReLUNeuron)
label_enm = SoftmaxLossLayer(net, ip2_enm, 1, 3)

sgd = SGD(100, 0.1)
solve(sgd, net)
