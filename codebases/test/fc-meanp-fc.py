'''
    Testcase: fc-meanp-fc.py
       1. FC layer with tanh activation
       2. Pooling layer with mean activation
       3. FC layer with tanh activation
       4. softmax layer
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm = LibsvmDataLayer(net, \
                    '../datasets/iris-scale-train.libsvm',  \
                    '../datasets/iris-scale-test.libsvm', 1, 4, 3)
ip1_enm = FullyConnectedLayer(net, data_enm, 1, 20, FCNeuron)
ip2_enm = PoolingLayer(net, ip1_enm, 1, 5, MeanPoolingNeuron, 1, 4)
ip3_enm = FullyConnectedLayer(net, ip2_enm, 1, 10, FCNeuron)
label_enm = SoftmaxLossLayer(net, ip3_enm, 1, 3)

sgd = SGD(100, 0.1)
solve(sgd, net)
