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
    _name = LibsvmDataLayer(_net, _train_file, _test_file, _dim_x, _dim_y, _nLabels)

@template
def template_SoftmaxLossLayer():
    _name = SoftmaxLossLayer(_net, _prev, _dim_x, _dim_y)

@template
def template_ConvolutionLayer():
    _name = ConvolutionLayer(_net, _prev, _dim_x, _dim_y, _Neuron, _ker_dim_x, _ker_dim_y)

@template
def template_PoolingLayer():
    _name = PoolingLayer(_net, _prev, _dim_x, _dim_y, _Neuron, _pool_dim_x, _pool_dim_y)

@template
def template_Ensemble():
    _net.add_ensemble(_cur_enm)

@template
def template_NewEnsembleShareWeights():
    _enm = Ensemble(_dim_x, _dim_y, _TYPE, share_weights=_share_weights)

@template
def template_NewEnsembleNoShareWeights():
    _enm = Ensemble(_dim_x, _dim_y, _TYPE)

new_ensemble_templates = [
    template_NewEnsembleShareWeights(),
    template_NewEnsembleNoShareWeights()
]

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
        template_PoolingLayer(),
        template_ConvolutionLayer(),
        template_SoftmaxLossLayer()
]

"""
Templates for computation programming paradigm
"""

@template
def template_for_backward_adj():
    for _i in self.backward_adj:
        _body

@template
def template_for(range):
    for _i in range(_N):
        _body

@template
def template_for_range(range):
    for _i in range(len(_array)):
        _body

for_templates = [ template_for_range("range"), template_for_range("xrange") ]

@template
def template_dp(target, varname):
    for _prev in self.backward_adj:
        target.varname += _A[_i][_j] * _B[_i][_j] 

@template
def template_fp_dp():
    for _prev in self.backward_adj:
        _C += _A[_i][_j] * _B[_i][_j] 


@template
def template_fp_sum():
    for _prev in self.backward_adj:
        _C += _A[_i][_j] 

@template
def template_bp_axpy():
    for _prev in self.backward_adj:
        _C[_i][_j] += _scalar * _B[_i][_j]

@template
def template_bp_scalar_prod():
    for _prev in self.backward_adj:
        _A += _alpha * _B[_i][_j]

@template
def template_asgn(field):
    self.field = _exp


