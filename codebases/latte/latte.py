import compiler, inspect
import compiler.ast
from abc import abstractmethod

# def neuron(neuron_subtype):
#     def neuron_inner():
#         source = inspect.getsource(neuron_subtype)
#         module = compiler.parse(source)
#         return module
#     return neuron_inner

#  _   _      _                      _     
# | \ | | ___| |___      _____  _ __| | __ 
# |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ / 
# | |\  |  __/ |_ \ V  V / (_) | |  |   <  
# |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\ 
#                                          
class Network(object):
    """
        Network: Container for the whole graph network.
                 All operations: create_ensemble, create_layer, etc
                 should be initiated by the network
    """
    def __init__(self):
        super(Network, self).__init__()

    def create_ensemble(self, neuron_type, num_neurons):
        ensemble = Ensemble(neuron_type, num_neurons)
        return ensemble

    def add_connections(self, ensemble_source, ensemble_sink, connections):
        layer = Layer(ensemble_source, ensemble_sink)
        # TODO: extract forward and backward function here
        forward  = layer.get_forward_ast()
        backward = layer.get_backward_ast()
        return layer

#  _   _                             
# | \ | | ___ _   _ _ __ ___  _ __   
# |  \| |/ _ \ | | | '__/ _ \| '_ \  
# | |\  |  __/ |_| | | | (_) | | | | 
# |_| \_|\___|\__,_|_|  \___/|_| |_| 
#                                    
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

#  _____                          _     _       
# | ____|_ __  ___  ___ _ __ ___ | |__ | | ___  
# |  _| | '_ \/ __|/ _ \ '_ ` _ \| '_ \| |/ _ \ 
# | |___| | | \__ \  __/ | | | | | |_) | |  __/ 
# |_____|_| |_|___/\___|_| |_| |_|_.__/|_|\___| 
#                                               
class Ensemble(object):
    """
        Ensemble: A group of neurons of same level
    """
    def __init__(self, neuron_type, num_neurons):
        super(Ensemble, self).__init__()
        self.neuron_type = neuron_type
        self.neurons = [ neuron_type() ] * num_neurons

#  _                           
# | |    __ _ _   _  ___ _ __  
# | |   / _` | | | |/ _ \ '__| 
# | |__| (_| | |_| |  __/ |    
# |_____\__,_|\__, |\___|_|    
#             |___/            
#
class Layer(object):

    """
        Layer: connection manager between two ensembles
    """

    def __init__(self, source, sink):
        super(Layer, self).__init__()
        self.ensemble_source = source
        self.ensemble_sink   = sink

    def get_forward_ast(self):
        source = inspect.getsource(neuron_subtype.forward)
        module = compiler.parse(source)
        return module

    def get_backward_ast(self):
        source = inspect.getsource(neuron_subtype.backward)
        module = compiler.parse(source)
        return module
