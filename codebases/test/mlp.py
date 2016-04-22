'''
    Testcase: Multi-layer Perceptron 
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm = LibsvmDataLayer(net, \
                    '../../datasets/iris-scale-train.libsvm',  \
                    '../../datasets/iris-scale-test.libsvm', 1, 4)
ip1_enm = FullyConnectedLayer(net, data_enm, 1, 20, Neuron)
ip2_enm = FullyConnectedLayer(net, ip1_enm, 1, 10, Neuron)
label_enm = SoftmaxLossLayer(net, ip2_enm, 1, 3)

sgd = SGD(10, 0.1)
solve(sgd, net)
