'''
    Standard library for Latte Programming Model
'''
import sys


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

# TODO: how to xaiver initialize
def Xaiver_init (M, N):
    
    return [[]]

def add_connection (net, prev_enm, cur_enm, mappings):
    # update adjacency lists
    for i, indices in mappings:
        assert 0 <= i and i < prev_enm.get_size()
        prev_enm[i].forward_adj = [ cur_enm[j] for j in indices ]
        for j in indices: cur_enm[j].backward_adj.append(prev_enm[i])
    # add weights
    net.add_weights(prev_enm, cur_enm, mappings)
    return

''' 
    libsvm format: label fea_id:f_val 
    Note that fea_id is 1-based 
'''
def read_libsvm (file_name, nFeatures, nLabels):
    fread = open(file_name, "r")
    features = []
    labels = []
    for line in fread:
        fields = line.split()
        fea = [ 0.0 ] * nFeatures
        for i in range(1, len(fields)):
            pair = fields[i].split(":")
            fea[int(pair[0])-1] = float(pair[1])
        features.append(fea)
        labels.append(int(fields[0]))
    fread.close()
    return features, labels

def LibsvmDataLayer(net, train_file, test_file, nFeatures, nLabels):
    # read data files
    train_features, train_labels = read_libsvm(train_file, nFeatures, nLabels)
    test_features, test_labels  = read_libsvm(test_file, nFeatures, nLabels)
    net.set_datasets(train_features, train_labels, test_features, test_features)
    '''
    for x in train_features: print x
    for x in train_labels: print x
    sys.exit(0)
    '''
    # construct data and label ensembles
    data_enm = Ensemble(nFeatures, DataNeuron)
    net.set_data_ensemble(data_enm)
    return data_enm, nLabels

def FullyConnectedLayer(net, prev_enm, N, TYPE):
    # construct a new ensemble
    M = prev_enm.get_size()
    cur_enm = Ensemble(N, TYPE)
    cur_enm.set_inputs_dim (1, M)
    # enforce connections
    mappings = {}
    for i in len(M): mappings.update(i, [j for j in range(len(N))])
    add_connection (net, prev_enm, cur_enm, mappings)
    net.add_ensemble (cur_enm)
    return cur_enm

def SoftmaxLossLayer(net, prev_enm, nLabels):
    # TODO: 
    label_enm = Ensemble(nLabels, SoftmaxNeuron)
    return 

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
        # innder product of inputs and weights

        # put output value to the inputs of next layer

        pass

    def backward(self):

        pass

# TODO:
class DataNeuron(Neuron):
    def __init__(self):
        pass

# TODO:
class SoftmaxNeuron(Neuron):
    def __init__(self):
        pass

class Ensemble:
    def __init__(self, N, TYPE):
        self.ensemble_id = allocate_ensemble_id()
        self.size = N
        self.neurons = [TYPE() for i in range(len(N))]
        return  

    def __eq__(self, other):
        return self.ensemble_id == other.ensemble_id

    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.neurons)
        return self.neurons[idx]

    def get_size(self): return self.size

    def set_inputs_dim(self, nrows, ncols):
        for neuron in self.neurons:
            neuron.inputs = [] 
            neuron.grad_inputs = []
            for i in range(nrows):
                neuron.inputs.append([0.0] * ncols)
                neuron.grad_inputs.append([0.0] * ncols)
        return 

class Network:
    def __init__(self):
        self.network_id = allocate_network_id()
        self.ensembles = []
        self.weights = {}
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

    def __eq__(self, other):
        return self.network_id == other.network_id

    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.ensembles)
        return self.ensembles[i]

    def get_ensembles(self): return self.ensembles

    def add_ensemble(self, enm):
        assert isinstance(enm, Ensemble) and len(self.ensembles) >= 2
        self.ensembles.append(enm)

    def set_data_ensemble(self, data_enm):
        assert len(self.ensembles) == 0 # must be empty ensembles
        self.ensembles = [data_enm]
    
    def set_datasets (self, train_fea, train_labels, test_fea, test_labels):
        self.train_features = train_fea
        self.train_labels = train_labels
        self.test_features = test_fea
        self.test_labels = test_labels

    def add_weights (self, prev_enm, cur_enm, mappings):
        tmp_weights = Xaiver_init(prev_enm.get_size(), cur_enm.get_size())
        self.weights.update({(prev_enm, cur_enm):tmp_weights})

#class WeightedNeuron(Neuron):

if __name__ == "__main__":
    # TODO: ADD Unit testing of standard library for latte HERE
    pass
