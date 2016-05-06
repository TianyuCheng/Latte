#include "Latte.h"

//using namespace std;



int main (int argn, char** argv) { 


// create neural networks 
Network net;

// create ensembles used in neural networks
Ensemble data_enm(28, 28, NULL); net.add_ensemble(&data_enm);
Ensemble ip1_enm(100, 100, &data_enm); net.add_ensemble(&ip1_enm);
Ensemble ip2_enm(50, 50, &ip1_enm); net.add_ensemble(&ip2_enm);
Ensemble label_enm(1, 10, &ip2_enm); net.add_ensemble(&label_enm);

// allocating memory for specific fields of data_enm
float* data_enm_grad_activation = init_mkl_mat(28, 28);
float* data_enm_output = init_mkl_mat(28, 28);
float* data_enm_grad_output = init_mkl_mat(28, 28);
// allocating memory for specific fields of ip1_enm
float* ip1_enm_grad_activation = init_mkl_mat(100, 100);
float* ip1_enm_output = init_mkl_mat(100, 100);
vector<vector<float*>> ip1_enm_weights (100, vector<float*>(100, NULL));
vector<vector<float*>> ip1_enm_grad_weights (100, vector<float*>(100, NULL));
float* ip1_enm_grad_output = init_mkl_mat(100, 100);
// allocating memory for specific fields of ip2_enm
float* ip2_enm_grad_activation = init_mkl_mat(50, 50);
float* ip2_enm_output = init_mkl_mat(50, 50);
vector<vector<float*>> ip2_enm_weights (50, vector<float*>(50, NULL));
vector<vector<float*>> ip2_enm_grad_weights (50, vector<float*>(50, NULL));
float* ip2_enm_grad_output = init_mkl_mat(50, 50);
// allocating memory for specific fields of label_enm
float* label_enm_grad_activation = init_mkl_mat(1, 10);
float* label_enm_output = init_mkl_mat(1, 10);
vector<vector<float*>> label_enm_weights (1, vector<float*>(10, NULL));
vector<vector<float*>> label_enm_grad_weights (1, vector<float*>(10, NULL));
float* label_enm_grad_output = init_mkl_mat(1, 10);

// initialize weights of layers 
init_weights_mats(ip1_enm_weights, 28, 28, true); 
init_weights_mats(ip2_enm_weights, 100, 100, true); 
init_weights_mats(label_enm_weights, 50, 50, true); 
init_weights_mats(ip1_enm_grad_weights, 28, 28, false); 
init_weights_mats(ip2_enm_grad_weights, 100, 100, false); 
init_weights_mats(label_enm_grad_weights, 50, 50, false); 

// load mnist data
vector<float*> train_features, test_features;
vector<int> train_labels, test_labels;
read_mnist("../datasets/mnist-train.csv", train_features, train_labels, 28, 28, 10);
read_mnist("../datasets/mnist-test.csv", test_features, test_labels, 28, 28, 10);
assert (train_features.size() == train_labels.size());
assert (test_features.size() == test_labels.size());
vector<int> shuffle_index;
generate_shuffle_index(shuffle_index, train_features.size());

// solve block
for ( int iter = 0 ; iter < 3 ; iter = iter + 1 ) {
    cout << "iteration: " << iter << endl;

for ( int si = 0 ; si < train_features.size() ; si = si + 1 ) {

int data_idx = shuffle_index[si];
sgemm_copy (data_enm_output, train_features[data_idx], 28*28);
vector<vector<int>> cur_label (1, vector<int>(10, 0));
cur_label[0][train_labels[data_idx]] = 1;

for (int x = 0; x < 100; x += 1) {
for (int y = 0; y < 100; y += 1) {
(*(ip1_enm_output+x*100+y)) = 0.0;
sgemm_dp((ip1_enm_output+x*100+y), ip1_enm_weights[x][y], data_enm_output, 784);
(*(ip1_enm_grad_activation+x*100+y)) = (1 - pow(tanh((*(ip1_enm_output+x*100+y))), 2));
(*(ip1_enm_output+x*100+y)) = tanh((*(ip1_enm_output+x*100+y)));
}
}

for (int x = 0; x < 50; x += 1) {
for (int y = 0; y < 50; y += 1) {
(*(ip2_enm_output+x*50+y)) = 0.0;
sgemm_dp((ip2_enm_output+x*50+y), ip2_enm_weights[x][y], ip1_enm_output, 10000);
(*(ip2_enm_grad_activation+x*50+y)) = (1 - pow(tanh((*(ip2_enm_output+x*50+y))), 2));
(*(ip2_enm_output+x*50+y)) = tanh((*(ip2_enm_output+x*50+y)));
}
}

for (int x = 0; x < 1; x += 1) {
for (int y = 0; y < 10; y += 1) {
(*(label_enm_output+x*10+y)) = 0.0;
sgemm_dp((label_enm_output+x*10+y), label_enm_weights[x][y], ip2_enm_output, 2500);
(*(label_enm_output+x*10+y)) = exp((*(label_enm_output+x*10+y)));
}
}

// annotate for loss layer
float sumover = 0.0;
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 10; y++) {
		sumover += *(label_enm_output+x*10+y);
	}
}
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 10; y++) {
		*(label_enm_output+x*10+y) = *(label_enm_output+x*10+y) / sumover;
	}
}

for (int x = 0; x < 1; x += 1) {
for (int y = 0; y < 10; y += 1) {
(*(label_enm_grad_output+x*10+y)) = ((*(label_enm_output+x*10+y)) - cur_label[x][y]);
sgemm_axpy(ip2_enm_grad_output, (*(label_enm_grad_output+x*10+y)), label_enm_weights[x][y], 2500);
sgemm_axpy(label_enm_grad_weights[x][y], (*(label_enm_grad_output+x*10+y)), ip2_enm_output, 2500);
}
}

for (int x = 0; x < 50; x += 1) {
for (int y = 0; y < 50; y += 1) {
(*(ip2_enm_grad_output+x*50+y)) = ((*(ip2_enm_grad_output+x*50+y)) * (*(ip2_enm_grad_activation+x*50+y)));
sgemm_axpy(ip1_enm_grad_output, (*(ip2_enm_grad_output+x*50+y)), ip2_enm_weights[x][y], 10000);
sgemm_axpy(ip2_enm_grad_weights[x][y], (*(ip2_enm_grad_output+x*50+y)), ip1_enm_output, 10000);
}
}

for (int x = 0; x < 100; x += 1) {
for (int y = 0; y < 100; y += 1) {
(*(ip1_enm_grad_output+x*100+y)) = ((*(ip1_enm_grad_output+x*100+y)) * (*(ip1_enm_grad_activation+x*100+y)));
sgemm_axpy(ip1_enm_grad_weights[x][y], (*(ip1_enm_grad_output+x*100+y)), data_enm_output, 784);
}
}

// weights_update for ip1_enm
for (int x = 0; x < 100; x++) {
	for (int y = 0; y < 100; y++) {
		sgemm_axpy(ip1_enm_weights[x][y], -0.1, ip1_enm_grad_weights[x][y], 28*28);
		sgemm_zeros(ip1_enm_grad_weights[x][y], 28*28);
	}
}
		sgemm_zeros(ip1_enm_grad_output, 100*100);

// weights_update for ip2_enm
for (int x = 0; x < 50; x++) {
	for (int y = 0; y < 50; y++) {
		sgemm_axpy(ip2_enm_weights[x][y], -0.1, ip2_enm_grad_weights[x][y], 100*100);
		sgemm_zeros(ip2_enm_grad_weights[x][y], 100*100);
	}
}
		sgemm_zeros(ip2_enm_grad_output, 50*50);

// weights_update for label_enm
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 10; y++) {
		sgemm_axpy(label_enm_weights[x][y], -0.1, label_enm_grad_weights[x][y], 50*50);
		sgemm_zeros(label_enm_grad_weights[x][y], 50*50);
	}
}
		sgemm_zeros(label_enm_grad_output, 1*10);


} // end of data instances traversal
} // end of iterative traversal

// test block
vector<int> preds;
for ( int data_idx = 0 ; data_idx < test_features.size() ; data_idx = data_idx + 1 ) {
int tid = 0;

sgemm_copy (data_enm_output, test_features[data_idx], 28*28);
vector<vector<int>> cur_label (1, vector<int>(10, 0));
cur_label[0][test_labels[data_idx]] = 1;
for (int x = 0; x < 100; x += 1) {
for (int y = 0; y < 100; y += 1) {
(*(ip1_enm_output+x*100+y)) = 0.0;
sgemm_dp((ip1_enm_output+x*100+y), ip1_enm_weights[x][y], data_enm_output, 784);
(*(ip1_enm_grad_activation+x*100+y)) = (1 - pow(tanh((*(ip1_enm_output+x*100+y))), 2));
(*(ip1_enm_output+x*100+y)) = tanh((*(ip1_enm_output+x*100+y)));
}
}

for (int x = 0; x < 50; x += 1) {
for (int y = 0; y < 50; y += 1) {
(*(ip2_enm_output+x*50+y)) = 0.0;
sgemm_dp((ip2_enm_output+x*50+y), ip2_enm_weights[x][y], ip1_enm_output, 10000);
(*(ip2_enm_grad_activation+x*50+y)) = (1 - pow(tanh((*(ip2_enm_output+x*50+y))), 2));
(*(ip2_enm_output+x*50+y)) = tanh((*(ip2_enm_output+x*50+y)));
}
}

for (int x = 0; x < 1; x += 1) {
for (int y = 0; y < 10; y += 1) {
(*(label_enm_output+x*10+y)) = 0.0;
sgemm_dp((label_enm_output+x*10+y), label_enm_weights[x][y], ip2_enm_output, 2500);
(*(label_enm_output+x*10+y)) = exp((*(label_enm_output+x*10+y)));
}
}

// annotate for loss layer in testing stage
int pred = argmax (label_enm_output, 1*10);
preds.push_back(pred);

}
// evaluate the accuracy performance
evaluate(preds, test_labels);

// deallocating memory for specific fields of data_enm
mkl_free(data_enm_grad_activation);
mkl_free(data_enm_output);
mkl_free(data_enm_grad_output);
// deallocating memory for specific fields of ip1_enm
mkl_free(ip1_enm_grad_activation);
mkl_free(ip1_enm_output);
free_weights_mats(ip1_enm_weights);
free_weights_mats(ip1_enm_grad_weights);
mkl_free(ip1_enm_grad_output);
// deallocating memory for specific fields of ip2_enm
mkl_free(ip2_enm_grad_activation);
mkl_free(ip2_enm_output);
free_weights_mats(ip2_enm_weights);
free_weights_mats(ip2_enm_grad_weights);
mkl_free(ip2_enm_grad_output);
// deallocating memory for specific fields of label_enm
mkl_free(label_enm_grad_activation);
mkl_free(label_enm_output);
free_weights_mats(label_enm_weights);
free_weights_mats(label_enm_grad_weights);
mkl_free(label_enm_grad_output);

}
