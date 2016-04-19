#ifndef LATTE_H
#define LATTE_H
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

// forward declaration for Latte classes
class Network;
class Neuron;
class Ensemble;
class Solver;
class SGDSolver;
class Connection;

void read_libsvm(vector<vector<float> > &features, vector<int> &labels, string &filename, int n_features, int &n_labels);
void shared_variable_analsyis();
void Xaiver_initialize();
void add_connection(Network& net, Ensemble& enm1, Ensemble& enm2, Connection &connection);

// Ensemble* LibsvmDataLayer(Network &net, string train_file, string test_file, int &n_features, int n_labels);
// Ensemble* FullyConnectedLayer(Network &net, Ensemble &prev_ensemble, int N);
// Ensemble* SoftmaxLossLayer(Network &net, Ensemble &prev_ensemble, int n_labels);

typedef struct Index {
    int r = 1;
    int c;
} Dim;

/**
 * Connection
 * Functor for connection mappings from ensemble to ensemble
 * */
class Connection
{
public:
    virtual Index operator() (Index index) = 0;
};

/**
 * Neuron class
 * Base class for Neuron subtyping
 * */
class Neuron
{
public:
    // constructor and destructor
    Neuron(Ensemble &ensemble, int pos_x, int pos_y) : x(pos_x), y(pos_y) {
    }
    virtual ~Neuron() {
    }

    // initialization functions
    void init_inputs_dim(int dim_x, int dim_y, int prev_enm_size);
    void init_grad_inputs_dim(int dim_x, int dim_y);
    
    // forward and backward propagation functions
    void forward();
    void backward();
private:
    int x;
    int y;
};

/**
 * Ensemble class
 * */
class Ensemble
{
public:
    // constructor and destructor
    Ensemble(Dim s) : size(s) {
        neurons.resize(size.r * size.c);
        // all neurons constructions are independent
        #pragma omp parallel for
        for (int i = 0; i < neurons.size(); ++i)
            neurons[i] = new Neuron(*this, i / s.r, i % s.c);
    }
    virtual ~Ensemble() {
        // all neurons destructions are independent
        #pragma omp parallel for
        for (int i = 0; i < neurons.size(); ++i)
            delete neurons[i];
    }
    int get_size() { return neurons.size(); }
    void set_forward_adj(Connection &forward_adj);
    void set_backward_adj(Connection &backward_adj);
private:
    Dim size;
    vector<Neuron*> neurons;
};

/**
 * Network class
 * */
class Network
{
public:
    // constructor and destructor
    Network();
    virtual ~Network();

    Ensemble& create_ensemble(Dim dim);
    const vector<int>& load_data_instance(int idx);

    // getters called for data loading
    vector<vector<float>> & get_train_features() { return train_features; }
    vector<vector<float>> & get_test_features() { return test_features; }
    vector<int> & get_train_labels() { return train_labels; }
    vector<int> & get_test_labels() { return test_labels; }

private:
    vector<Ensemble> ensembles;
    // data
    vector<vector<float>> train_features;
    vector<vector<float>> test_features;
    vector<int> train_labels;
    vector<int> test_labels;
};

/**
 * Base Solver class
 * */
class Solver
{
public:
    // constructor and destructor
    Solver() { }
    virtual ~Solver();
    // need to override this abstract function: solve
    virtual void solve(Network &net) = 0;
};

/**
 * Stochastic Gradient Descent
 * Solver for Deep Neural Network
 * */
class SGDSolver : public Solver
{
public:
    SGDSolver(int iter) : iterations(iter) {
    }
    virtual ~SGDSolver ();
    void solve(Network &net);
private:
    int iterations;
};

#endif
