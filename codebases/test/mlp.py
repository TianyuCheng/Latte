'''
    Testcase: Multi-layer Perceptron 
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm, nLabels = LibsvmDataLayer(net, \
                    "../../datasets/iris-scale-train.libsvm",  \
                    "../../datasets/iris-scale-test.libsvm", 4, 3)
ip1_enm = FullyConnectedLayer(net, data_enm, 20, 20, Neuron)
ip2_enm = FullyConnectedLayer(net, ip1_enm, 10, 10, Neuron)
label_enm = SoftmaxLossLayer(net, ip2_enm, nLabels)

sgd = SGD(10, 0.1)
solve(sgd, net)
