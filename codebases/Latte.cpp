#include "Latte.h"

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

Ensemble* LibsvmDataLayer(Network &net, string train_file, string test_file, 
                          int n_features, int &n_labels) {
    vector<vector<float>> &train_features = net.get_train_features();
    vector<vector<float>> &test_features = net.get_test_features();
    vector<int> &train_labels = net.get_train_labels();
    vector<int> &test_labels = net.get_test_labels();

    //read_libsvm(train_features, train_labels, train_file, n_features, n_labels);
    //read_libsvm(test_features, test_labels, test_file, n_features, n_labels);

    // TODO: this function is not finished yet
    // How do we arrange the data, SoA directly, or AoS -> SoA?
    return nullptr;
}

Ensemble* FullyConnectedLayer(Network &net, Ensemble &prev_ensemble, int N) {
    return nullptr;
}

Ensemble* SoftmaxLossLayer(Network &net, Ensemble &prev_ensemble, int n_labels) {

    return nullptr;
}

void add_connection(Network& net, Ensemble& enm1, Ensemble& enm2, Connection &connection) {

    return;
}

void shared_variable_analsyis() {

    return;
}


/*
Ensemble& Network::create_ensemble(Dim dim) {
    ensembles.push_back(Ensemble(dim));
    return ensembles.back();
}
*/


int ITERATIONS = 1000;
void solve (Network& network, Solver& solver) {
    
    // 1. mapping to shared memory region
    // INPUT: network architecture
    // OUTPUT: compact memory allocation of computing neurons 
    // shared_variable_analysis(); 
    // 2. initialize the connection (weights) parameter of network 
    // 3. 
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 4. forward routine through pattern matching

        // 5. backward routine through pattern matching 
        // (solver like sgd is used here..)

        // 6. update weight
    }
    // 7. evaluate accuracy and timing performance
    // evaluate ();
    
}

int main (int argn, char** argv) {
    vector<float*> features; vector<int> labels;
    read_libsvm("../datasets/iris-scale-train.libsvm", features, labels, 1, 4, 3);
    cout << "read_libsvm: ../datasets/iris-scale-train.libsvm" << endl;
    for (int data_idx = 0; data_idx < features.size(); data_idx++) {
        cout << labels[data_idx]  << " ";
        for (int i = 0; i < 1; i ++) {
            for (int j = 0; j < 4; j ++) {
                cout << *(features[data_idx]+ i*4+j) << " " ;
            }
        }
        cout << endl;
    }

    float* A = init_mkl_mat(5, 5);
    Xaiver_initialize(A, 25, 30);
    cout << "Mat A: " << endl; sgemm_print(A, 5, 5);
    cout << "Mat A: " << endl;

    float* B = init_mkl_mat(5, 5);
    Xaiver_initialize(B, 25, 30);
    cout << "Mat B: " << endl; sgemm_print(B, 5, 5);
    
    float* C = init_mkl_mat(25, 25);
    int m = 1, n = 1, k = 25;
    float alpha = 1.0, beta = 0.0;
    int lda = k, ldb = k, ldc = 1;
    sgemm_dp (C, A, B, k);
    cout << "Mat C: " << *C  << endl;
    cout << "Mat C: " << *(C+1)  << endl;

    float scalar = 5.0;
    sgemm_axpy (B, scalar, A, 25);
    cout << "Mat B: " << endl; sgemm_print(B, 5, 5);
    sgemm_zeros(B, 25);
    cout << "Mat B: " << endl; sgemm_print(B, 5, 5);
    sgemm_copy (B, A, 25);
    cout << "Mat B: " << endl; sgemm_print(B, 5, 5);

    mkl_free(C);
    mkl_free(B);
    mkl_free(A);
#if 0
    Network net1;
    Network net2;
    Network net3;
    cout << "[net1] " << net1.get_string() << endl;
    cout << "[net2] " << net2.get_string() << endl;
    cout << "[net3] " << net3.get_string() << endl;

    Neuron neuron1;
    Neuron neuron2;
    Neuron neuron3;
    cout << "[neuron1] " << neuron1.get_string() << endl;
    cout << "[neuron2] " << neuron2.get_string() << endl;
    cout << "[neuron3] " << neuron3.get_string() << endl;

    /* TODO: more unit tests here */
#endif
}
