#ifndef LATTE_H
#define LATTE_H
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

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
void add_connection(Network& net, Ensemble& enm1, Ensemble& enm2, Connection &connection);

// Ensemble* LibsvmDataLayer(Network &net, string train_file, string test_file, int &n_features, int n_labels);
// Ensemble* FullyConnectedLayer(Network &net, Ensemble &prev_ensemble, int N);
// Ensemble* SoftmaxLossLayer(Network &net, Ensemble &prev_ensemble, int n_labels);

float min (vector<float> vec) {
    assert(vec.size() > 0 && "empty vector input.");
    float min_value = vec[0];
    for (int i = 1; i < vec.size(); i++) 
        if (vec[i] < min_value) min_value = vec[i];
    return min_value;
}
float max (vector<float> vec) {
    assert(vec.size() > 0 && "empty vector input.");
    float max_value = vec[0];
    for (int i = 1; i < vec.size(); i++) 
        if (vec[i] > max_value) max_value = vec[i];
    return max_value;
}
int argmin (vector<float> vec) {
    assert(vec.size() > 0 && "empty vector input.");
    int min_index = 0;
    float min_value = vec[0];
    for (int i = 1; i < vec.size(); i++) 
        if (vec[i] < min_value) {
            min_index = i;
            min_value = vec[i];
        }
    return min_index;
}
int argmax (vector<float> vec) {
    assert(vec.size() > 0 && "empty vector input.");
    int max_index = 0;
    float max_value = vec[0];
    for (int i = 1; i < vec.size(); i++) 
        if (vec[i] > max_value) {
            max_index = i;
            max_value = vec[i];
        }
    return max_index;
}
void Xaiver_initialize (float* mat, int n_j, int n_jp) {
    float high = sqrt(6.0 / (n_j+n_jp)), low = -1.0 * high;
    random_device rd;
    default_random_engine generator( rd() );
    uniform_real_distribution<float> distribution(0.9, 1.0);
    // uniform_real_distribution<float> distribution(low, high);
    for (int i = 0; i < n_j; i ++) *(mat+i) = distribution(generator);
}
float* init_mkl_mat (int dim_x, int dim_y) {
    return (float*) mkl_malloc ( dim_x*dim_y*sizeof(float), 64);;
}
void init_weights_mats (vector<vector<float*>>& mat, int prev_dim_x, int prev_dim_y) {
    assert(mat.size() > 0 && "mat.size should be greater than 0");
    int dim_x = mat.size(), dim_y = mat[0].size();
    int n_j = prev_dim_x * prev_dim_y, n_jp = dim_x * dim_y;
    for (int i = 0; i < dim_x; i ++) {
        for (int j = 0; j < dim_y; j ++) {
            mat[i][j] = init_mkl_mat(prev_dim_x, prev_dim_y);
            Xaiver_initialize(mat[i][j], n_j, n_jp);
        }
    }
}
void free_weights_mats (vector<vector<float*>>& mat) {
    assert(mat.size() > 0 && "mat.size should be greater than 0");
    int dim_x = mat.size(), dim_y = mat[0].size();
    for (int i = 0; i < dim_x; i ++) 
        for (int j = 0; j < dim_y; j ++) 
            mkl_free(mat[i][j]);
}

void evaluate (vector<int>& preds, vector<int>& labels) {
    assert (preds.size() == labels.size() && "preds and labels has different size.");
    int numCorrect = 0, totalItems = preds.size();
    for (int i = 0 ; i < totalItems; i ++) {
        if (preds[i] == labels[i])
            numCorrect ++;
    }
    float accuracy = 1.0 * numCorrect / totalItems;
    cout << "numCorrect: " << numCorrect 
         << ", Accuracy: " << accuracy
         << endl;
    return ;
}

/* Dot product of two matrices (float *) */
void sgemm (float* C, float* A, float* B, int size) {
    cblas_sgemm ( 
            CblasColMajor, // 
            CblasTrans,    // op(A)
            CblasNoTrans,  // op(B)
            1, 1, size,    // m, n, k
            1.0, A, size,  // alpha, A, lda
            B, size,       //        B, ldb
            0.0, C, 1      // beta,  C, ldc
            );   
}

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
    int dim_x, dim_y; 
    Ensemble *prev, *next;
    // constructor and destructor
    Ensemble(int dim_x, int dim_y, Ensemble* prev) {
        this->dim_x = dim_x; 
        this->dim_y = dim_y; 
        this->prev = prev;
        if (prev != NULL) prev->next = this;
    }
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
    vector<Ensemble*> ensembles;
    // constructor and destructor
    Network() { }
    virtual ~Network() {}

    void add_ensemble(Ensemble* enm) { 
        this->ensembles.push_back(enm);
    }
    // Ensemble& create_ensemble(Dim dim);
    const vector<int>& load_data_instance(int idx);

    // getters called for data loading
    vector<vector<float>> & get_train_features() { return train_features; }
    vector<vector<float>> & get_test_features() { return test_features; }
    vector<int> & get_train_labels() { return train_labels; }
    vector<int> & get_test_labels() { return test_labels; }

private:
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
    SGDSolver(int iter, float step) : iterations(iter) {
    }
    virtual ~SGDSolver ();
    void solve(Network &net);
private:
    int iterations;
};

#endif
