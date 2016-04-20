from ast_matcher import template

"""
Templates for input scripts
"""

@template
def template_Network():
    _name = Network() 

@template
def template_FullyConnectedLayer():
    _name = FullyConnectedLayer(_net, _enm, _N1, _N2, _Neuron)

@template
def template_LibsvmDataLayer():
    _name, _nLabels = LibsvmDataLayer(_net, _train, _test, _nFeatures, _nLabels)

@template
def template_SoftmaxLossLayer():
    _name = SoftmaxLossLayer(_net, _enm, _nLabels)

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
def template_axpy_body1():
    _y[_i] += _alpha * _x[_i]

@template
def template_axpy_body2():
    _y[_i] += _x[_i] * _alpha

@template
def template_axpy_body3():
    _y[_i] = y[_i] + _alpha * _x[_i]

@template
def template_axpy_body4():
    _y[_i] = y[_i] + _x[_i] * _alpha

@template
def template_axpy_body5():
    _y[_i] = _alpha * _x[_i] + y[_i]

@template
def template_axpy_body6():
    _y[_i] = _x[_i] * _alpha + y[_i]

axpy_body_templates = [
        template_axpy_body1(),
        template_axpy_body2(),
        template_axpy_body3(),
        template_axpy_body4(),
        template_axpy_body5(),
        template_axpy_body6()
]

def match_axpy(ast):
    for template in for_templates:
        if not template.matchall(ast): continue
        for match in template.matches:
            body = match['_body']
            for axpy in axpy_body_templates:
                if not axpy.matchall(body): continue
                for axpy_match in axpy.matches:
                    print axpy_match
