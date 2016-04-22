from ast_matcher import template

"""
Templates for input scripts
"""

@template
def template_Network():
    _name = Network() 

@template
def template_FullyConnectedLayer():
    _name = FullyConnectedLayer(_net, _prev, _dim_x, _dim_y, _Neuron)

@template
def template_LibsvmDataLayer():
    _name = LibsvmDataLayer(_net, _train, _test, _dim_x, _dim_y)

@template
def template_SoftmaxLossLayer():
    _name = SoftmaxLossLayer(_net, _prev, _dim_x, _dim_y)

@template
def template_Ensemble():
    _net.add_ensemble(_cur_enm)

@template
def template_SGD():
    _name = SGD(_iter, _step)

@template
def template_add_connection():
    add_connection(_net, _prev_enm, _cur_enm, _mappings)

''' list of templates for layers '''
layer_templates = [
        template_LibsvmDataLayer(),
        template_FullyConnectedLayer(),
        template_SoftmaxLossLayer()
]

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

for_templates = [ template_for_range(), template_for_xrange() ]

@template
def template_axpy(range):
    for _i in range(len(_array)):
        _y[_i] = _alpha * _x[_i] + _y[i]
