#ifndef LATTE_H
#define LATTE_H
#include <string>
#include <cstring>
#include <vector>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>    // std::random_shuffle
#include <mkl.h>

#include <boost/random/mersenne_twister.hpp>
// #include <boost/random/random_device.hpp>
#include <boost/random/uniform_real_distribution.hpp>

using namespace std;

// forward declaration for Latte classes
class Network;
class Neuron;
class Ensemble;
class Solver;
class SGDSolver;
class Connection;

void shared_variable_analsyis();
void add_connection(Network& net, Ensemble& enm1, Ensemble& enm2, Connection &connection);

// Ensemble* LibsvmDataLayer(Network &net, string train_file, string test_file, int &n_features, int n_labels);
// Ensemble* FullyConnectedLayer(Network &net, Ensemble &prev_ensemble, int N);
// Ensemble* SoftmaxLossLayer(Network &net, Ensemble &prev_ensemble, int n_labels);

inline float min (vector<float> vec) {
    assert(vec.size() > 0 && "empty vector input.");
    float min_value = vec[0];
    for (int i = 1; i < vec.size(); i++) 
        if (vec[i] < min_value) min_value = vec[i];
    return min_value;
}
inline float max (float* vec, int size) {
    assert(size > 0 && "empty vector input.");
    float max_value = *vec;
    for (int i = 1; i < size; i++) 
        if (*(vec+i) > max_value) 
            max_value = *(vec+i);
    return max_value;
}

inline int argmin (vector<float> vec) {
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
inline int argmax (float* vec, int size) {
    assert(size > 0 && "empty vector input.");
    int max_index = 0;
    float max_value = *vec;
    for (int i = 1; i < size; i++) {
        if (!(*(vec+i) < INFINITY)) return i;
        if (*(vec+i) > max_value) {
            max_index = i;
            max_value = *(vec+i);
        }
    }
    return max_index;
}
void Xaiver_initialize (float* mat, int n_j, int n_jp) {
    // n_j  = 4; n_jp = 3;
    float high = sqrt(6.0 / (n_j+n_jp)), low = -1.0 * high;
    // boost::random::random_device generator;
    // boost::random::mt19937 generator ();
    std::random_device rd;
    std::mt19937 generator(rd());
    // generator.seed(std::time(0));
    boost::random::uniform_real_distribution<float> distribution(low, high);
    for (int i = 0; i < n_j; i ++) *(mat+i) = distribution(generator);
}
float* init_mkl_mat (int dim_x, int dim_y) {
    return (float*) mkl_malloc ( dim_x*dim_y*sizeof(float), 64);
}
void init_weights_mats (vector<vector<float*>>& mat, int prev_dim_x, int prev_dim_y, bool rand) {
    assert(mat.size() > 0 && "mat.size should be greater than 0");
    int dim_x = mat.size(), dim_y = mat[0].size();
    int n_j = prev_dim_x * prev_dim_y, n_jp = dim_x * dim_y;
    for (int i = 0; i < dim_x; i ++) {
        for (int j = 0; j < dim_y; j ++) {
            mat[i][j] = init_mkl_mat(prev_dim_x, prev_dim_y);
            if (rand) Xaiver_initialize(mat[i][j], n_j, n_jp);
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
    int numCorrect = 0, numTotalItems = preds.size();
    for (int i = 0 ; i < numTotalItems; i ++) {
        if (preds[i] == labels[i]) {
            numCorrect ++;
            cout << "[T]" ;
        } else 
            cout << "[F]";
        cout << " Pred: " << preds[i] << ", target: " << labels[i] << endl;
    }
    float accuracy = 1.0 * numCorrect / numTotalItems;
    cout << "numCorrect: " << numCorrect 
         << ", Accuracy: " << accuracy
         << endl;
    return ;
}

/* Dot product of two matrices (float *) */
inline void sgemm_dp (float* C, float* A, float* B, int size) {
    // A, B, C are all rolling into vector in memory
    // C = alpha * A * B + beta * C
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
/* scalar product of two matrices (float *) */
inline void sgemm_axpy (float* C, float scalar, float* B, int size) {
    // A, B, C are all rolling into vector in memory
    // C = alpha * scalar * B + beta * C
    cblas_sgemm ( 
            CblasColMajor,     // 
            CblasTrans,        // op(A)
            CblasTrans,        // op(B)
            1, size, 1,        // m, n, k
            1.0, &scalar, 1,   // alpha, A, lda
            B, size,           //        B, ldb
            1.0, C, 1          // beta,  C, ldc
            );   
}
inline void sgemm_zeros (float* C, int size) {
    // A, B, C are all rolling into vector in memory
    // C = [ 0.0 ] * size
    float scalar = 0.0;
    cblas_sgemm ( 
            CblasColMajor,     // 
            CblasTrans,        // op(A)
            CblasTrans,        // op(B)
            1, size, 1,        // m, n, k
            0.0, &scalar, 1,   // alpha, A, lda
            &scalar, size,     //        B, ldb
            0.0, C, 1          // beta,  C, ldc
            );   
}
inline void sgemm_copy (float* sink, float* source, int n) {
    // A, B, C are all rolling into vector in memory
    float scalar = 0.0;
    cblas_scopy (n, source, 1, sink, 1);
}
inline void sgemm_print (float* C, int dim_x, int dim_y) {
    for (int i = 0; i < dim_x; i ++) {
        for (int j = 0; j < dim_y; j ++) {
            cout << *(C+dim_y*i+j) << " ";
        }
        cout << endl;
    }
}

void read_mnist(string filename, vector<float*> &features, vector<int> &labels,
        size_t fea_dim_x, size_t fea_dim_y, int n_classes) {

    // read file and check the validity of the input file
    ifstream in(filename, std::ifstream::in);
    if (!in.good()) {
        cerr << "Error read file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    // getting the first line for feature dimension
    string line;
    getline(in, line);
    size_t num_features = count(line.begin(), line.end(), ',');
    assert (num_features % fea_dim_x == 0 && "features not in rectangular shape");
    size_t n_fea_dim_x = num_features / fea_dim_x;
    cout << fea_dim_x << ","  << n_fea_dim_x << endl;

    int label;
    char comma;
    // read all data points
    while (getline(in, line)) {
        istringstream ss(line);
        features.push_back(init_mkl_mat(n_fea_dim_x, fea_dim_x));
        float *data_point = features.back();

        // first read the label
        ss >> label;
        labels.push_back(label);

        // start reading features
        for (int i = 0; i < n_fea_dim_x; i++) {
            for (int j = 0; j < fea_dim_x; j++) {
                ss >> comma >> data_point[i * fea_dim_x + j];
            }
        } // end of reading features

    } // end of all reading while loop

#if 0       // DBEUG the first record
    float *data_point = features[0];
    for (int i = 0; i < n_fea_dim_x; i++) {
        for (int j = 0; j < fea_dim_x; j++) {
            float value = data_point[i * fea_dim_x + j];
            cout << data_point[i * fea_dim_x + j] << ", " ;
        }
    } // end of reading features
    cout << endl;
#endif

    // start normalizing features
    // #pragma omp parallel for
    for (int i = 0; i < features.size(); i++) {
        float *data_point = features[i];
        for (int i = 0; i < n_fea_dim_x; i++) {
            for (int j = 0; j < fea_dim_x; j++) {
                float value = data_point[i * fea_dim_x + j];
                assert(value >= 0 && value <= 255);
                data_point[i * fea_dim_x + j] = value / 255.0f;
            }
        } // end of reading features
    }
    // end normalizing features
}

void read_libsvm(string filename, vector<float*> &features, vector<int> &labels, 
        int fea_dim_x, int fea_dim_y, int n_classes) {

    // read file and check the validity of the input file
    ifstream in(filename, std::ifstream::in);
    if (!in.good()) {
        cerr << "Error read file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    // read data points from the dataset
    for (string line; getline(in, line); ) {
        features.push_back(init_mkl_mat(fea_dim_x, fea_dim_y));
        float *data_point = features.back();

        // find substring separated by space
        line += ":";      // avoid hitting stirng::npos
        size_t start = 0, end = 0, curr = 0, prev = 0;
        for (int i = 0; (end = line.find(" ", start)) != string::npos;i++) {
            string str = line.substr(start, end - start);
            start = end + 1;

            // the first token is the label
            if (i == 0) {
                labels.push_back(stoi(str)-1);
            }
            else {
                size_t slice = str.find(":", 0);
                assert(slice != string::npos && "cannot parse file");
                curr = stoi(str.substr(0, slice)) - 1;
                data_point[curr] = stof(str.substr(slice + 1, str.length()));

                if (curr - prev > 1) memset(&data_point[curr], 0, curr-prev);
                prev = curr;
            }
        } // end of single data point processing
    } // end of all data points processing
}


void generate_shuffle_index(vector<int> &shuffle_index, int size) {
    for (int i = 0; i < size; i ++) 
        shuffle_index.push_back(i);

    random_shuffle ( shuffle_index.begin(), shuffle_index.end() );

    /*
        for (int i = 0; i < train_features.size(); i ++) 
            cout << "shuffle_index: " << shuffle_index[i] << endl;
            */
}

timespec time_diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

typedef struct Index {
    int r = 1;
    int c;
} Dim;

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
        // neurons.resize(size.r * size.c);
        // all neurons constructions are independent
        // #pragma omp parallel for
        // for (int i = 0; i < neurons.size(); ++i)
        //     neurons[i] = new Neuron(*this, i / s.r, i % s.c);
    }
    virtual ~Ensemble() {
        // all neurons destructions are independent
        // #pragma omp parallel for
        // for (int i = 0; i < neurons.size(); ++i)
        //     delete neurons[i];
    }
    // int get_size() { return neurons.size(); }
    // void set_forward_adj(Connection &forward_adj);
    // void set_backward_adj(Connection &backward_adj);
private:
    Dim size;
    // vector<Neuron*> neurons;
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

#endif
