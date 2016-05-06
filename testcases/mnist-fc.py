'''
    Testcase: Multi-layer Perceptron 
'''
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm = MnistDataLayer(net, \
                    '../datasets/mnist-train.csv',  \
                    '../datasets/mnist-test.csv', 28, 28, 10)
ip1_enm = FullyConnectedLayer(net, data_enm, 50, 50, FCNeuron)
label_enm = SoftmaxLossLayer(net, ip1_enm, 1, 10)

sgd = SGD(3, 0.1)
solve(sgd, net)
