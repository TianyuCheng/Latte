'''
    Testcase: Multi-layer Perceptron 
'''

from libLatte import *

net = Network()
data_enm, label_enm = binLibsvmDataLayer(net, )
ip1_enm = FullyConnectedLayer(net, data_enm, 20, Neuron)
ip2_enm = FullyConnectedLayer(net, ip1_enm, 10, Neuron)
SigmoidLossLayer(net, ip2_enm, label_enm)
