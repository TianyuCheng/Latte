from ast_matcher import template

"""
Templates for input scripts
"""

@template
def template_Network():
    _name = Network() 

@template
def template_FullyConnectedLayer():
    _name = FullyConnectedLayer(_net, _enm, _N, _Neuron)

@template
def template_LibsvmDataLayer():
    _data_enm, _nLabels = LibsvmDataLayer(_net, _train, _test, _nFeatures, _nLabels)

@template
def template_SoftmaxLossLayer():
    _label_enm = SoftmaxLossLayer(_net, _enm, _nLabels)

@template
def template_Ensemble():
    _net.add_ensemble(_cur_enm)

@template
def template_SGD():
    _name = SGD(_iter, _step)

@template
def template_add_connection():
    add_connection(_net, _prev_enm, _cur_enm, _mappings)

"""
Templates for computation programming paradigm
"""

@template
def template_for_range():
    for _i in range(len(_array)):
        _body

@template
def template_for_xrange():
    for _i in xrange(len(_array)):
        _body
