'''
    Standard library for Latte Programming Model
'''


ensemble_id_counter = 0
neuron_id_counter = 0
network_id_counter = 0
def allocate_neuron_id ():
    assigned_id = neuron_id_counter
    global neuron_id_counter
    neuron_id_counter += 1
    return assigned_id
def allocate_ensemble_id ():
    assigned_id = ensemble_id_counter
    global ensemble_id_counter
    ensemble_id_counter += 1
    return assigned_id
def allocate_network_id ():
    assigned_id = network_id_counter
    global network_id_counter
    network_id_counter += 1
    return assigned_id

# TODO: how to xaiver initialize and represent weights?
def add_connection (net, prev_enm, cur_enm, mappings):
    for i, indices in mappings:
        assert 0 <= i and i < prev_enm.get_size()
        prev_enm[i].forward_adj = [ cur_enm[j] for j in indices ]
        for j in indices: cur_enm[j].backward_adj.append(prev_enm[i])
    return

def binLibsvmDataLayer(net, train_file, test_file, dim):
    pass

def FullyConnectedLayer(net, prev_enm, N, TYPE):
    M = prev_enm.get_size()
    cur_enm = Ensemble(N, TYPE)
    cur_enm.set_inputs_dim (1, M)
    mappings = {}
    for i in len(M): mappings.update(i, [j for j in len(N)])
    add_connection (net, prev_enm, cur_enm, mappings)
    net.add_ensemble (cur_enm)
    return cur_enm

def SigmoidLossLayer():
    pass

class Neuron:
    def __init__(self):
        self.neuron_id = allocate_ensemble_id()
        self.inputs = [[]]
        self.grad_inputs = [[]]
        self.output = 0.0
        self.grad_output = 0.0

        self.forward_adj = []   # forward adjacency list
        self.backward_adj = []  # backward adjacency list
        return 

    def __eq__(self, other):
        return self.neuron_id == other.neuron_id

    def forward(self):
        pass

    def backward(self):
        pass

class Ensemble:
    def __init__(self, N, TYPE):
        self.ensemble_id = allocate_ensemble_id()
        self.size = N
        self.neurons = [TYPE() for i in len(N)]
        return  

    def __eq__(self, other):
        return self.ensemble_id == other.ensemble_id

    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.neurons)
        return self.neurons[idx]

    def get_size(self):
        return self.size

    def set_inputs_dim(self, nrows, ncols):
        for neuron in self.neurons:
            neuron.inputs = [] 
            neuron.grad_inputs = []
            for i in len(nrows):
                neuron.inputs.append([0.0 * ncols])
                neuron.grad_inputs.append([0.0 * ncols])
        return 

class Network:
    def __init__(self):
        self.network_id = allocate_network_id()
        self.ensembles = []
        pass

    def __eq__(self, other):
        return self.network_id == other.network_id

    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.ensembles)
        return self.ensembles[i]

    def get_ensembles(self):
        return self.ensembles

    def add_ensemble(self, enm):
        assert isinstance(enm, Ensemble)
        self.ensembles.append(enm)


#class WeightedNeuron(Neuron):


if __name__ == "__main__":
    # TODO: ADD Unit testing of standard library for latte HERE
    pass
