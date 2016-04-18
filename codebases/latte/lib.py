'''
    Standard library for Latte Programming Model
'''
import sys
import numpy as np


ensemble_id_counter = 0
neuron_id_counter = 0
network_id_counter = 0
def allocate_neuron_id ():
    global neuron_id_counter
    assigned_id = neuron_id_counter
    neuron_id_counter += 1
    return assigned_id
def allocate_ensemble_id ():
    global ensemble_id_counter
    assigned_id = ensemble_id_counter
    ensemble_id_counter += 1
    return assigned_id
def allocate_network_id ():
    global network_id_counter
    assigned_id = network_id_counter
    network_id_counter += 1
    return assigned_id

def Xaiver_weights_init (dim_x, dim_y, prev_enm_size):
    cur_enm_size = dim_x * dim_y;
    high = np.sqrt( 6.0 / (prev_enm_size + cur_enm_size) )
    low = -1.0 * high 
    result = []
    for i in range(dim_x):
        result.append([ np.random.uniform(low, high) for j in range(dim_y) ])
    #print result
    return result

def add_connection (net, prev_enm, cur_enm, mappings):
    # update adjacency lists
    for i, indices in mappings.iteritems():
        assert 0 <= i and i < prev_enm.get_size()
        prev_enm[i].forward_adj = [ cur_enm[j] for j in indices ]
        for j in indices: cur_enm[j].backward_adj.append(prev_enm[i])
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
    cur_enm.set_backward_adj(prev_enm)
    prev_enm.set_forward_adj(cur_enm)
    cur_enm.set_inputs_dim (1, M)
    # enforce connections
    mappings = {}
    for i in range(M): mappings.update({i:[j for j in range(N)]})
    add_connection (net, prev_enm, cur_enm, mappings)
    net.add_ensemble (cur_enm)
    return cur_enm

def SoftmaxLossLayer(net, prev_enm, nLabels):
    # TODO: 
    label_enm = Ensemble(nLabels, SoftmaxNeuron)
    return 

class Neuron:
    def __init__(self, enm, pos_x, pos_y):
        # management info
        self.neuron_id = allocate_ensemble_id()
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.enm = enm
        # data info
        self.weights     = [[]]
        self.inputs      = [[]]
        self.grad_inputs = [[]]
        self.output      = 0.0
        self.grad_output = 0.0
        # architecture info
        self.forward_adj  = []   # forward adjacency list
        self.backward_adj = []  # backward adjacency list
        return 

    def __eq__(self, other):
        return self.neuron_id == other.neuron_id

    def init_dim (self, dim_x, dim_y, prev_enm_size):
        self.inputs      = [ [0.0] * dim_y ] * dim_x
        self.grad_inputs = [ [0.0] * dim_y ] * dim_x
        self.weights     = Xaiver_weights_init (dim_x, dim_y, prev_enm_size)

    def forward(self):
        # innder product of inputs and weights
        assert len(forward_adj) > 0, "No forward adjacency element. "
        dp_result = 0.0
        for i in len(self.inputs):
            for j in len(self.inputs[0]):
                dp_result = self.weights[i][j] * self.inputs[i][j]
        self.output = np.tanh(dp_result)
        # put output value to the inputs of next layer
        for next_neuron in forward_adj:
            next_neuron.inputs[self.pos_x, self.pos_y] = self.output
        pass

    def backward(self):
        
        pass

class DataNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)

    def forward(self):
        # remember to load input feature to data neuron before forward propa
        assert len(forward_adj) > 0, "No forward adjacency element. "
        for next_neuron in forward_adj:
            next_neuron.inputs[self.pos_x, self.pos_y] = self.output

    def backward(self):
        pass # no backward propagation for data neuron

# TODO:
class SoftmaxNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.label = None
    
    def forward(self):
        dp_result = 0.0
        for i in len(self.inputs):
            for j in len(self.inputs[0]):
                dp_result = self.weights[i][j] * self.inputs[i][j]
        self.output = e**dp_result

    # NOTE: remember to invoke this annotate() and before backward
    def annotate(self):
        size = self.enm.get_size()
        divisor = sum( [ self.enm.neurons[i].output for i in range(size) ] )
        self.output = self.output / divisor

    def backward(self):
        diff = self.label - self.output
        pass

class Ensemble:
    def __init__(self, N, TYPE):
        self.ensemble_id = allocate_ensemble_id()
        self.size = N
        # NOTE: currently only allow 1-d ensemble
        self.neurons = [TYPE(self, 1, i) for i in range(N)]
        self.prev_adj_enm = None
        self.next_adj_enm = None
        return  

    def __eq__(self, other): 
        return self.ensemble_id == other.ensemble_id

    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.neurons)
        return self.neurons[idx]

    def get_size(self): return self.size
    def set_forward_adj (self, enm):  self.next_adj_enm = enm
    def set_backward_adj (self, enm): self.prev_adj_enm = enm
    def set_inputs_dim(self, dim_x, dim_y):
        prev_enm_size = self.prev_adj_enm.get_size()
        for neuron in self.neurons: 
            neuron.init_dim (dim_x, dim_y, prev_enm_size)

class Network:
    def __init__(self):
        self.network_id = allocate_network_id()
        self.ensembles = []
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
        assert isinstance(enm, Ensemble) and len(self.ensembles) >= 1
        self.ensembles.append(enm)

    def set_data_ensemble(self, data_enm):
        assert len(self.ensembles) == 0 # must be empty ensembles
        self.ensembles = [data_enm]
    
    def set_datasets (self, train_fea, train_labels, test_fea, test_labels):
        self.train_features = train_fea
        self.train_labels = train_labels
        self.test_features = test_fea
        self.test_labels = test_labels

#class WeightedNeuron(Neuron):

if __name__ == "__main__":
    # TODO: ADD Unit testing of standard library for latte HERE
    pass
