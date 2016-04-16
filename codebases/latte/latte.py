import compiler, inspect
import compiler.ast

def neuron(neuron_subtype):
    def neuron_inner():
        source = inspect.getsource(neuron_subtype)
        module = compiler.parse(source)
        return module
    return neuron_inner

ensemble = "Ensemble"
network = "Network"
forward = "forward"
backward = "backward"
