#ifndef LATTE_H
#define LATTE_H
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cassert>

using namespace std;

// forward declaration for Latte classes
class Network;
class Neuron;
class Ensemble;
class Solver;
class SGDSolver;
class Connection;

class Neuron
{
public:
    // constructor and destructor
    Neuron(Ensemble &ensemble, int pos_x, int pos_y);
    virtual ~Neuron();

    // initialization functions
    void init_inputs_dim(int dim_x, int dim_y, int prev_enm_size);
    void init_grad_inputs_dim(int dim_x, int dim_y);
    
    // forward and backward propagation functions
    virtual void forward() = 0;
    virtual void backward() = 0;
private:
    int pos_x;
    int pos_y;
};

/**
 * Ensemble class
 * The template refers to the type of 
 * Neuron residing in the ensemble
 * */
class Ensemble
{
public:
    // constructor and destructor
    Ensemble(int size);
    Ensemble(int row, int col);
    virtual ~Ensemble();

    int get_size() { return neurons.size(); }
    void set_forward_adj(Connection &forward_adj);
    void set_backward_adj(Connection &backward_adj);
private:
    std::vector<Neuron*> neurons;
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

    void add_ensemble(Ensemble &ensemble);
    void set_data_ensemble(Ensemble &data_ensemble);
    void set_datasets(vector<vector<int>> &train_feature, 
                      vector<int> &train_labels,
                      vector<vector<int>> &test_feature, 
                      vector<int> &test_labels);
    const vector<int>& load_data_instance(int idx);
private:
    vector<Ensemble> ensembles;
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
