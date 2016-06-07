'''
    Created by Loc after the semester
    Testcase: fc-conv-fc-higgs.py
       1. FC layer with tanh activation
       2. Convolution layer with mean activation
       3. FC layer with tanh activation
       4. softmax layer
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm = LibsvmDataLayer(net,\
                    '../datasets/higgs-train.libsvm',\
                    '../datasets/higgs-test.libsvm', 1, 28, 2)
ip1_enm = FullyConnectedLayer(net, data_enm, 100, 100, FCNeuron)
ip2_enm = PoolingLayer(net, ip1_enm, 50, 50, MeanPoolingNeuron, 2, 2)
ip3_enm = FullyConnectedLayer(net, ip2_enm, 50, 50, FCNeuron)
label_enm = SoftmaxLossLayer(net, ip3_enm, 1, 10)

sgd = SGD(100, 0.1)
solve(sgd, net)
