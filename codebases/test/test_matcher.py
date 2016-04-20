#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.abspath(".."))

import inspect, compiler
from latte.ast_matcher import *
from latte.templates import *

def test_for_range():
    tmpl = template_for_range()
    ast = compiler.parse("""
for i in range(len(inputs[0])):
    print inputs[0][i]
    """)
    ast = ast_remove_module(ast)
    matched = tmpl.match(ast)
    print "Match?", matched
    if matched:
        for key, value in tmpl.wildcard.iteritems():
            print "%s:\t%s" % (key, value)

def test_matchall():
    ast = compiler.parse("""
import os, sys
sys.path.append(os.path.abspath(".."))

from latte.lib import *

net = Network()
data_enm, nLabels = LibsvmDataLayer(net, \
                    "../../datasets/iris-scale-train.libsvm",  \
                    "../../datasets/iris-scale-test.libsvm", 4, 3)
ip1_enm = FullyConnectedLayer(net, data_enm, 20, Neuron)
ip2_enm = FullyConnectedLayer(net, ip1_enm, 10, Neuron)
label_enm = SoftmaxLossLayer(net, ip2_enm, nLabels)

sgd = SGD(10, 0.1)
solve(sgd, net)
    """)
    tmpl = template_Network()
    matched = tmpl.matchall(ast)
    print "========================="
    print "Match:", matched
    for match in tmpl.matches:
        for key, value in match.iteritems():
            print "%s:\t%s" % (key, value)

    tmpl = template_FullyConnectedLayer()
    matched = tmpl.matchall(ast)
    print "========================="
    print "Match:", matched
    for match in tmpl.matches:
        print "----------------------"
        for key, value in match.iteritems():
            print "%s:\t%s" % (key, value)

    tmpl = template_LibsvmDataLayer()
    matched = tmpl.matchall(ast)
    print "========================="
    print "Match:", matched
    for match in tmpl.matches:
        print "----------------------"
        for key, value in match.iteritems():
            print "%s:\t%s" % (key, value)

    tmpl = template_SoftmaxLossLayer()
    matched = tmpl.matchall(ast)
    print "========================="
    print "Match:", matched
    for match in tmpl.matches:
        print "----------------------"
        for key, value in match.iteritems():
            print "%s:\t%s" % (key, value)

def test_axpy():
    ast = compiler.parse("""
for i in range(len(inputs[0])):
    y[i] += x[i] * 0.1

for i in range(len(inputs[0])):
    y[i] += 0.1 * x[i]

for i in range(len(inputs[0])):
    y[i] = y[i] + 0.1 * x[i]

for i in range(len(inputs[0])):
    y[i] = x[i] * 0.1 + y[i]
    """)
    match_axpy(ast)

if __name__ == "__main__":
    # test_for_range()
    # test_matchall()
    test_axpy()
