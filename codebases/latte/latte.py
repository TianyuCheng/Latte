import compiler, inspect
import compiler.ast
from abc import abstractmethod

# def neuron(neuron_subtype):
#     def neuron_inner():
#         source = inspect.getsource(neuron_subtype)
#         module = compiler.parse(source)
#         return module
#     return neuron_inner

class Neuron(object):
    """
        Neuron: Base type for neuron.
                Each neuron subtype should inherit
                Neuron type for common fields
    """
    def __init__(self):
        super(Neuron, self).__init__()
        # basic requirements
        self.value           = None     # type Float32
        self.gradient        = None     # type Float32
        self.inputs          = None     # type Vector<Vector>
        self.inputs_gradient = None     # type Vector<Vector>

    @abstractmethod
    def forward(self):
        """ describe the forward propagation to next ensemble, must be user-defined """
        return

    @abstractmethod
    def backward(self):
        """ describe the backward propagation to next ensemble, must be user-defined """
        return
