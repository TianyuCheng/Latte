'''
    Standard library for Latte Programming Model
'''
import sys
import numpy as np
import math
import random 
import time

def Xaiver_weights_init (dim_x, dim_y, cur_enm_size):
    prev_enm_size = dim_x * dim_y;
    high = np.sqrt( 6.0 / (prev_enm_size + cur_enm_size) )
    low = -1.0 * high 
    result = []
    for i in range(dim_x):
        result.append([ np.random.uniform(low, high) for j in range(dim_y) ])
    #print result
    return result

def add_connection (net, prev_enm, cur_enm, mappings):
    #TODO deprecated? will we end up using this?

    # update adjacency lists
    for i, indices in mappings.iteritems():
        assert 0 <= i and i < prev_enm.get_size()
        prev_enm[0][i].forward_adj = [ cur_enm[0][j] for j in indices ]
        for j in indices: cur_enm[0][j].backward_adj.append(prev_enm[0][i])
    return

''' 
    libsvm format: label fea_id:f_val 
    Note that fea_id is 1-based 
'''
def read_libsvm (file_name, fea_dim_x, fea_dim_y):
    fread = open(file_name, "r")
    features = []
    labels = []

    for line in fread:
        fields = line.split()
        fea = [ 0.0 ] * fea_dim_y 

        for i in range(1, len(fields)):
            pair = fields[i].split(":")
            fea[int(pair[0])-1] = float(pair[1])

        features.append(fea)
        labels.append(int(fields[0]))

    fread.close()
    return features, labels

def FullyConnectedLayer(net, prev, dim_x, dim_y, TYPE):
    # construct a new ensemble
    cur_enm = Ensemble(dim_x, dim_y, TYPE)
    add_connection(net, prev, cur_enm, lambda x, y: \
         [ (i,j) for i in range(prev.dim_x) for j in range(prev.dim_y) ])
    net.add_ensemble (cur_enm)
    return cur_enm

def ConvolutionLayer(net, prev, dim_x, dim_y, TYPE, ker_dim_x, ker_dim_y):
    '''
    1d constraint:
        assert dim_x == ker_dim_x == prev_dim_x 
        assert dim_y + ker_dim_y == prev_dim_y
    2d constraint:
        assert dim_x + ker_dim_x == prev_dim_x 
        assert dim_y + ker_dim_y == prev_dim_y
    '''
    cur_enm = Ensemble(dim_x, dim_y, TYPE, share_weights=True)
    add_connection(net, prev, cur_enm, lambda x, y: \
         [ (i,j) for i in range(x, x+ker_dim_x) \
                 for j in range(y, y+ker_dim_y) ])
    net.add_ensemble (cur_enm)

def PoolingLayer(net, prev, dim_x, dim_y, TYPE, pool_dim_x, pool_dim_y):
    '''
        assert prev_dim_x % pool_dim_x == 0
        assert prev_dim_y % pool_dim_y == 0
        assert prev_dim_x / pool_dim_x == dim_x
        assert prev_dim_y / pool_dim_y == dim_y
    '''
    cur_enm = Ensemble(dim_x, dim_y, TYPE)
    add_connection(net, prev, cur_enm, lambda x, y: \
         [ (i,j) for i in range(x*pool_dim_x, (x+1)*pool_dim_x) \
                 for j in range(y*pool_dim_y, (y+1)*pool_dim_y) ])
    net.add_ensemble (cur_enm)

def SoftmaxLossLayer(net, prev, dim_x, dim_y):
    label_enm = Ensemble(1, nLabels, SoftmaxNeuron)
    return FullyConnectedLayer(net, prev, dim_x, dim_y, SoftmaxNeuron)

def One2OneLayer(net, prev, dim_x, dim_y, TYPE):
    cur_enm = Ensemble(dim_x, dim_y, TYPE)
    add_connection(net, prev, cur_enm, lambda x, y: \
         [ (i,j) for i in range(x, x+1) \
                 for j in range(y, y+1) ])
    return cur_enm

class Neuron:
    def __init__(self, enm, pos_x, pos_y):
        # management info
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.prev_dim_x = 0
        self.prev_dim_y = 0
        self.enm = enm
        # data info
        self.inputs       = [[]]  # fp: restore activation of neurons in last layer
        self.grad_inputs  = [[]]  # bp: restore error of last layer
        self.output       = 0.0   # fp: restore activation value of self
        self.grad_output  = 0.0   # bp: restore error of self

        self.grad_activation = 0.0
        # architecture info
        self.forward_adj  = []  # forward adjacency list
        self.backward_adj = []  # backward adjacency list
        return 

    def __eq__(self, other):
        return self.neuron_id == other.neuron_id

    def init_inputs_dim (self, dim_x, dim_y):
        self.prev_dim_x = dim_x
        self.prev_dim_y = dim_y
        self.inputs      = [ [0.0] * dim_y ] * dim_x
        self.weights     = Xaiver_weights_init (dim_x, dim_y, self.enm.get_size())
        self.grad_inputs = [ [0.0] * dim_y ] * dim_x
        self.grad_weights= [ [0.0] * dim_y ] * dim_x

    def forward(self): pass

    def backward(self): pass

    def clear_grad_weights(self):
        for i in range(self.prev_dim_x):
            for j in range(self.prev_dim_y):
                self.grad_weights[i][j] = 0.0 
        # clean up grad_output
        self.grad_output = 0.0

class FCNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.weights      = [[]]
        self.grad_weights = [[]]
        return 

    def forward(self):
        # innder product of inputs and weights
        assert len(self.forward_adj) > 0, "No forward adjacency element. "
        self.output = 0.0 
        for prev in self.backward_adj:
            self.output += self.weights[prev.pos_x][prev.pos_y] * self.inputs[prev.pos_x][prev.pos_y]
        # preset the gradient for back propagation
        self.grad_activation = (1 - np.tanh(self.output) ** 2 )
        # activation
        self.output = np.tanh(self.output) 

    def backward(self):
        self.grad_output = self.grad_output * self.grad_activation
        # backpropagate error
        for prev in self.backward_adj:
            prev.grad_output += self.grad_output * self.weights[prev.pos_x][prev.pos_y]
        # weights to update
        for prev in self.backward_adj:
            self.grad_weights[prev.pos_x][prev.pos_y] += self.grad_output * self.inputs[prev.pos_x][prev.pos_y]


class WeightedNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.weights      = [[]]
        self.grad_weights = [[]]

    def forward(self):
        self.output = 0.0 
        for prev in self.backward_adj:
            self.output += self.weights[prev.pos_x][prev.pos_y] * self.inputs[prev.pos_x][prev.pos_y]
        self.grad_activation = 1.0
        #for next_neuron in self.forward_adj:
        #    next_neuron.inputs[self.pos_x][self.pos_y] = self.output

    def backward(self): 
        self.grad_output = self.grad_output * self.grad_activation
        # backpropagate error
        for prev in self.backward_adj:
            prev.grad_output += self.grad_output * self.weights[prev.pos_x][prev.pos_y]
        # weights to update
        for prev in self.backward_adj:
            self.grad_weights[prev.pos_x][prev.pos_y] += self.grad_output * self.inputs[prev.pos_x][prev.pos_y]

class ReLUNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.weights      = [[]]
        self.grad_weights = [[]]

    def forward(self):
        # innder product of inputs and weights
        assert len(self.forward_adj) > 0, "No forward adjacency element. "
        self.output = 0.0
        for prev in self.backward_adj:
            self.output += self.weights[prev.pos_x][prev.pos_y] * self.inputs[prev.pos_x][prev.pos_y]
        self.grad_activation = 1.0 / (1 + np.exp(-1.0*self.output))  # logistic
        self.output = np.log(np.exp(self.output) + 1) # softplus function

    def backward(self):
        self.grad_output = self.grad_output * self.grad_activation
        # backpropagate error
        for prev in self.backward_adj:
            prev.grad_output += self.grad_output * self.weights[prev.pos_x][prev.pos_y]
        # weights to update
        for prev in self.backward_adj:
            self.grad_weights[prev.pos_x][prev.pos_y] += self.grad_output * self.inputs[prev.pos_x][prev.pos_y]

class MeanPoolingNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y, pool_dim_x, pool_dim_y)
        self.pool_dim_x = pool_dim_x
        self.pool_dim_y = pool_dim_y

    def forward(self):
        self.output = 0.0
        for prev in self.backward_adj:
            self.output += self.inputs[prev.pos_x][prev.pos_y]
        self.output = self.output / (self.pool_dim_x * self.pool_dim_y) 
        # preset the gradient for back propagation
        self.grad_activation = 1.0 / (self.pool_dim_x * self.pool_dim_y) 

    def backward(self):
        self.grad_output = self.grad_output * self.grad_activation
        # backpropagate error
        for prev in self.backward_adj:
            prev.grad_output += self.grad_output / (self.pool_dim_x * self.pool_dim_y) 
        
class DataNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
    
    def __claim__(self):
        self.output = 0.0

    def forward(self):
        pass

    def backward(self):
        pass # no backward propagation for data neuron

class SigmoidNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.weights      = [[]]
        self.grad_weights = [[]]

    def forward(self):
        pass

    def backward(self):
        pass 

class SoftmaxNeuron(Neuron):
    def __init__(self, enm, pos_x, pos_y):
        Neuron.__init__(self, enm, pos_x, pos_y)
        self.label = None
        self.weights      = [[]]
        self.grad_weights = [[]]
    
    def forward(self):
        self.output = 0.0 
        for prev in self.backward_adj:
            self.output += self.weights[prev.pos_x][prev.pos_y] * self.inputs[prev.pos_x][prev.pos_y]
        self.output = math.exp(self.output)

    # NOTE: remember to invoke this annotate() and before backward
    def annotate(self):
        size = self.enm.get_size()
        divisor = sum( [ self.enm.neurons[0][i].output for i in range(size) ] )
        self.output = self.output / divisor

    def backward(self):
        self.grad_output = self.output - self.label 
        # backpropagate error
        for prev in self.backward_adj:
            prev.grad_output += self.grad_output * self.weights[prev.pos_x][prev.pos_y]
        # weights to update
        for prev in self.backward_adj:
            self.grad_weights[prev.pos_x][prev.pos_y] += self.grad_output * self.inputs[prev.pos_x][prev.pos_y]


class Ensemble:
    def __init__(self, N1, N2, TYPE, share_weights=False):
        self.dim_x = N1
        self.dim_y = N2
        self.size = N1 * N2
        # NOTE: currently only allow 1-d ensemble
        self.neurons = [ [ TYPE(self, i, j) for j in range(N2) ] for i in range(N1) ]
        self.prev_adj_enm = None
        self.next_adj_enm = None
        return  

    def __eq__(self, other): 
        return self.ensemble_id == other.ensemble_id
    def __len__(self): return len(self.neurons)
    def __getitem__(self, idx):
        assert 0 <= idx and idx < len(self.neurons)
        return self.neurons[idx]

    def get_size(self): return self.size
    def set_forward_adj(self, enm):  self.next_adj_enm = enm
    def set_backward_adj(self, enm): self.prev_adj_enm = enm
    def set_inputs_dim(self, prev_dim_x, prev_dim_y):
        for i in range(self.dim_x): 
            for j in range(self.dim_y): 
                self.neurons[i][j].init_inputs_dim (prev_dim_x, prev_dim_y)

    def run_forward_propagate(self):
        for i in range(self.dim_x): 
            for j in range(self.dim_y): 
                self.neurons[i][j].forward()
    def run_backward_propagate(self):
        for i in range(self.dim_x): 
            for j in range(self.dim_y): 
                self.neurons[i][j].backward()

class Network:
    def __init__(self):
        self.ensembles = []
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

    def __eq__(self, other):
        return self.network_id == other.network_id

    def __getitem__(self, idx):
        assert idx < len(self.ensembles)
        return self.ensembles[idx]

    def get_ensembles(self): return self.ensembles

    def add_ensemble(self, enm):
        assert isinstance(enm, Ensemble) and len(self.ensembles) >= 1
        self.ensembles.append(enm)

    def set_data_ensemble(self, data_enm):
        assert len(self.ensembles) == 0 # must be empty ensembles
        self.ensembles = [data_enm]
    
    def set_datasets (self, train_fea, train_labels, test_fea, test_labels, shuffle=True):
        if shuffle:
            indexes = range(len(train_fea))
            random.seed(1)
            random.shuffle(indexes)
            self.train_features, self.train_labels = [], []
            for i in range(len(indexes)):
                self.train_features.append(train_fea[indexes[i]])
                self.train_labels.append(train_labels[indexes[i]])
        else:
            self.train_features = train_fea
            self.train_labels = train_labels

        #for i in range(len(train_fea)):
        #    print self.train_labels[i], self.train_features[i]
        self.test_features = test_fea
        self.test_labels = test_labels

    def load_data_instance(self, idx, train=True):
        if train: 
            features_mat = self.train_features
            labels_vec = self.train_labels
        else:
            features_mat = self.test_features
            labels_vec = self.test_labels

        dim_data = len(self.ensembles[0].neurons[0])
        for i in range(dim_data):
            self.ensembles[0].neurons[0][i].output = features_mat[idx][i]

        dim_label = len(self.ensembles[-1].neurons[0])
        for i in range(dim_label):
            if i == labels_vec[idx] - 1:
                self.ensembles[-1].neurons[0][i].label = 1
            else: 
                self.ensembles[-1].neurons[0][i].label = 0

class Solver:
    def __init__(self, iterations):
        self.iterations = iterations
        pass

    def solve(self):
        pass

class SGD(Solver):
    def __init__(self, iterations, step_size):
        Solver.__init__(self, iterations)
        self.alpha = step_size

    def update_weights(self, net):
        for enm in net.ensembles: 
            # print len(enm), len(enm[0])
            for i in range(enm.dim_x):
                for j in range(enm.dim_y):
                    for prev in enm[i][j].backward_adj:
                        # print i, j, prev.pos_x, prev.pos_y, len(enm[i][j].grad_weights), len(enm[i][j].grad_weights[0])
                        enm[i][j].weights[prev.pos_x][prev.pos_y] -= enm[i][j].grad_weights[prev.pos_x][prev.pos_y]
                        enm[i][j].clear_grad_weights()

    def solve(self, net):
        assert net.train_features is not None
        assert net.train_labels is not None
        assert net.test_features is not None
        assert net.test_labels is not None
        train_size = len(net.train_features)
        test_size = len(net.test_features)
        assert train_size == len(net.train_labels)
        assert test_size == len(net.test_labels)
        
        begin = time.time()
        for iter_count in range(self.iterations):
            for data_idx in range(train_size):
                net.load_data_instance(data_idx)
                for i in range(len(net.ensembles)): 
                    net[i].run_forward_propagate()
                for j in range(len(net[-1][0])):
                    net[-1][0][j].annotate()
                for i in range(len(net.ensembles)): 
                    net[i].run_backward_propagate()
                self.update_weights(net)
            elapse = time.time() - begin
            print "Iter: %d" % iter_count, "Time Elapse:", elapse, "seconds..."
        end = time.time()

        # performance evaluation
        preds = []
        for data_idx in range(test_size):
            net.load_data_instance(data_idx, train=False)
            for i in range(len(net.ensembles)): 
                net[i].run_forward_propagate()
            pred = np.argmax ([ out_neuron.output for out_neuron in net[-1].neurons[0]] )
            preds.append(pred)
        assert(len(preds) == test_size), "dimensionality of preds and test_size does not match"
        nCorrect = sum([preds[i] == net.test_labels[i]-1 for i in range(test_size)])
        for i in range(len(preds)):
            print "preds: ", preds[i]+1, ", target: ", net.test_labels[i]
        print "Accuracy:", 1.0 * nCorrect / test_size
        print "Total Time Cost:", end-begin, "seconds."

def solve(solver, net):
    assert isinstance(solver, Solver), "solve: solver argument is not type Solver"
    assert isinstance(net, Network), "solve: net argument is not type Network"
    solver.solve(net)

#class WeightedNeuron(Neuron):


if __name__ == "__main__":
    # TODO: ADD Unit testing of standard library for latte HERE
    pass
