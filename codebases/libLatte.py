'''
    Standard library for Latte Programming Model
'''



def add_connection ():
    pass

def binLibsvmDataLayer():
    pass

def FullyConnectedLayer(net, data, size):
    pass

def SigmoidLossLayer():
    pass

class Neuron:
    def __init__(self):
        inputs = [[]]
        grad_input = [[]]
        output = 0.0
        grad_output = 0.0
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Ensemble:
    def __init__(self):
        pass

class Network:
    def __init__(self):
        self.ensembles = []
        pass

    def get_ensemble(self, idx):
        assert 0 <= idx and idx < len(self.ensembles)
        return self.ensembles[i]

    def get_ensembles(self):
        return self.ensembles

#class WeightedNeuron(Neuron):


if __name__ == "__main__":
    # TODO: ADD Unit testing of standard library for latte HERE
    pass
