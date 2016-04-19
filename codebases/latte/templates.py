from ast_matcher import template

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
def template_for_range():
    for _i in range(len(_array)):
        _body
