#include "Latte.h"
#include <omp.h>

//using namespace std;

int main (int argn, char** argv) { 

// OMP Library Initialization Block
omp_set_num_threads(5);

// create neural networks 
Network net;

// create ensembles used in neural networks
Ensemble data_enm(1, 4, NULL); net.add_ensemble(&data_enm);
Ensemble ip1_enm(1, 20, &data_enm); net.add_ensemble(&ip1_enm);
Ensemble ip2_enm(1, 10, &ip1_enm); net.add_ensemble(&ip2_enm);
Ensemble label_enm(1, 3, &ip2_enm); net.add_ensemble(&label_enm);

// allocating memory for specific fields of data_enm
vector<float*> data_enm_grad_activation (5, NULL); 
for (int i = 0; i < 5; i ++) data_enm_grad_activation[i] = init_mkl_mat(1, 4);
vector<float*> data_enm_output (5, NULL); 
for (int i = 0; i < 5; i ++) data_enm_output[i] = init_mkl_mat(1, 4);
vector<float*> data_enm_grad_output (5, NULL); 
for (int i = 0; i < 5; i ++) data_enm_grad_output[i] = init_mkl_mat(1, 4);
// allocating memory for specific fields of ip1_enm
vector<float*> ip1_enm_grad_activation (5, NULL); 
for (int i = 0; i < 5; i ++) ip1_enm_grad_activation[i] = init_mkl_mat(1, 20);
vector<float*> ip1_enm_output (5, NULL); 
for (int i = 0; i < 5; i ++) ip1_enm_output[i] = init_mkl_mat(1, 20);
vector<vector<float*>> ip1_enm_weights (1, vector<float*>(20, NULL));
vector<vector<vector<float*>>> ip1_enm_grad_weights (5, vector<vector<float*>>(1, vector<float*>(20, NULL)));
vector<float*> ip1_enm_grad_output (5, NULL); 
for (int i = 0; i < 5; i ++) ip1_enm_grad_output[i] = init_mkl_mat(1, 20);
// allocating memory for specific fields of ip2_enm
vector<float*> ip2_enm_grad_activation (5, NULL); 
for (int i = 0; i < 5; i ++) ip2_enm_grad_activation[i] = init_mkl_mat(1, 10);
vector<float*> ip2_enm_output (5, NULL); 
for (int i = 0; i < 5; i ++) ip2_enm_output[i] = init_mkl_mat(1, 10);
vector<vector<float*>> ip2_enm_weights (1, vector<float*>(10, NULL));
vector<vector<vector<float*>>> ip2_enm_grad_weights (5, vector<vector<float*>>(1, vector<float*>(10, NULL)));
vector<float*> ip2_enm_grad_output (5, NULL); 
for (int i = 0; i < 5; i ++) ip2_enm_grad_output[i] = init_mkl_mat(1, 10);
// allocating memory for specific fields of label_enm
vector<float*> label_enm_grad_activation (5, NULL); 
for (int i = 0; i < 5; i ++) label_enm_grad_activation[i] = init_mkl_mat(1, 3);
vector<float*> label_enm_output (5, NULL); 
for (int i = 0; i < 5; i ++) label_enm_output[i] = init_mkl_mat(1, 3);
vector<vector<float*>> label_enm_weights (1, vector<float*>(3, NULL));
vector<vector<vector<float*>>> label_enm_grad_weights (5, vector<vector<float*>>(1, vector<float*>(3, NULL)));
vector<float*> label_enm_grad_output (5, NULL); 
for (int i = 0; i < 5; i ++) label_enm_grad_output[i] = init_mkl_mat(1, 3);

// initialize weights of layers 
init_weights_mats(ip1_enm_weights, 1, 4); 
init_weights_mats(ip2_enm_weights, 1, 20); 
init_weights_mats(label_enm_weights, 1, 10); 
for (int i = 0; i < 5; i ++) init_weights_mats(ip1_enm_grad_weights[i], 1, 4); 
for (int i = 0; i < 5; i ++) init_weights_mats(ip2_enm_grad_weights[i], 1, 20); 
for (int i = 0; i < 5; i ++) init_weights_mats(label_enm_grad_weights[i], 1, 10); 

// load libsvm data
vector<float*> train_features, test_features;
vector<int> train_labels, test_labels;
read_libsvm("../datasets/iris-scale-train.libsvm", train_features, train_labels, 1, 4, 3);
read_libsvm("../datasets/iris-scale-test.libsvm", test_features, test_labels, 1, 4, 3);
assert (train_features.size() == train_labels.size());
assert (test_features.size() == test_labels.size());
vector<int> shuffle_index;
generate_shuffle_index(shuffle_index, train_features.size());

// solve block
for ( int iter = 0 ; iter < 100 ; iter = iter + 1 ) {

#pragma omp for collapse(2) schedule(static, 1) private(tid, data_idx, cur_label, sumover)
for ( int si = 0 ; si < train_features.size() ; si = si + 1 ) {
int tid = omp_get_thread_num();

int data_idx = shuffle_index[si];
sgemm_copy (data_enm_output[tid], train_features[data_idx], 1*4);
vector<vector<int>> cur_label (1, vector<int>(3, 0));
cur_label[0][train_labels[data_idx]] = 1;

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 20; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip1_enm_output[tid]+x*20+y)) = 0.0;
sgemm_dp((ip1_enm_output[tid]+x*20+y), ip1_enm_weights[x][y], data_enm_output[tid], 4);
(*(ip1_enm_grad_activation[tid]+x*20+y)) = (1 - pow(tanh((*(ip1_enm_output[tid]+x*20+y))), 2));
(*(ip1_enm_output[tid]+x*20+y)) = tanh((*(ip1_enm_output[tid]+x*20+y)));
}
for (int _remain_y = 18; _remain_y < 2; _remain_y += 1) {
(*(ip1_enm_output[tid]+x*20+_remain_y)) = 0.0;
sgemm_dp((ip1_enm_output[tid]+x*20+_remain_y), ip1_enm_weights[x][_remain_y], data_enm_output[tid], 4);
(*(ip1_enm_grad_activation[tid]+x*20+_remain_y)) = (1 - pow(tanh((*(ip1_enm_output[tid]+x*20+_remain_y))), 2));
(*(ip1_enm_output[tid]+x*20+_remain_y)) = tanh((*(ip1_enm_output[tid]+x*20+_remain_y)));
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 10; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip2_enm_output[tid]+x*10+y)) = 0.0;
sgemm_dp((ip2_enm_output[tid]+x*10+y), ip2_enm_weights[x][y], ip1_enm_output[tid], 20);
(*(ip2_enm_grad_activation[tid]+x*10+y)) = (1 - pow(tanh((*(ip2_enm_output[tid]+x*10+y))), 2));
(*(ip2_enm_output[tid]+x*10+y)) = tanh((*(ip2_enm_output[tid]+x*10+y)));
}
for (int _remain_y = 9; _remain_y < 1; _remain_y += 1) {
(*(ip2_enm_output[tid]+x*10+_remain_y)) = 0.0;
sgemm_dp((ip2_enm_output[tid]+x*10+_remain_y), ip2_enm_weights[x][_remain_y], ip1_enm_output[tid], 20);
(*(ip2_enm_grad_activation[tid]+x*10+_remain_y)) = (1 - pow(tanh((*(ip2_enm_output[tid]+x*10+_remain_y))), 2));
(*(ip2_enm_output[tid]+x*10+_remain_y)) = tanh((*(ip2_enm_output[tid]+x*10+_remain_y)));
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 3; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(label_enm_output[tid]+x*3+y)) = 0.0;
sgemm_dp((label_enm_output[tid]+x*3+y), label_enm_weights[x][y], ip2_enm_output[tid], 10);
(*(label_enm_output[tid]+x*3+y)) = exp((*(label_enm_output[tid]+x*3+y)));
}
}
}

// annotate for loss layer
float sumover = 0.0;
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 3; y++) {
		sumover += *(label_enm_output[tid]+x*3+y);
	}
}
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 3; y++) {
		*(label_enm_output[tid]+x*3+y) = *(label_enm_output[tid]+x*3+y) / sumover;
	}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 3; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(label_enm_grad_output[tid]+x*3+y)) = ((*(label_enm_output[tid]+x*3+y)) - cur_label[x][y]);
sgemm_axpy(ip2_enm_grad_output[tid], (*(label_enm_grad_output[tid]+x*3+y)), label_enm_weights[x][y], 10);
sgemm_axpy(label_enm_grad_weights[tid][x][y], (*(label_enm_grad_output[tid]+x*3+y)), ip2_enm_output[tid], 10);
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 10; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip2_enm_grad_output[tid]+x*10+y)) = ((*(ip2_enm_grad_output[tid]+x*10+y)) * (*(ip2_enm_grad_activation[tid]+x*10+y)));
sgemm_axpy(ip1_enm_grad_output[tid], (*(ip2_enm_grad_output[tid]+x*10+y)), ip2_enm_weights[x][y], 20);
sgemm_axpy(ip2_enm_grad_weights[tid][x][y], (*(ip2_enm_grad_output[tid]+x*10+y)), ip1_enm_output[tid], 20);
}
for (int _remain_y = 9; _remain_y < 1; _remain_y += 1) {
(*(ip2_enm_grad_output[tid]+x*10+_remain_y)) = ((*(ip2_enm_grad_output[tid]+x*10+_remain_y)) * (*(ip2_enm_grad_activation[tid]+x*10+_remain_y)));
sgemm_axpy(ip1_enm_grad_output[tid], (*(ip2_enm_grad_output[tid]+x*10+_remain_y)), ip2_enm_weights[x][_remain_y], 20);
sgemm_axpy(ip2_enm_grad_weights[tid][x][_remain_y], (*(ip2_enm_grad_output[tid]+x*10+_remain_y)), ip1_enm_output[tid], 20);
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 20; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip1_enm_grad_output[tid]+x*20+y)) = ((*(ip1_enm_grad_output[tid]+x*20+y)) * (*(ip1_enm_grad_activation[tid]+x*20+y)));
sgemm_axpy(ip1_enm_grad_weights[tid][x][y], (*(ip1_enm_grad_output[tid]+x*20+y)), data_enm_output[tid], 4);
}
for (int _remain_y = 18; _remain_y < 2; _remain_y += 1) {
(*(ip1_enm_grad_output[tid]+x*20+_remain_y)) = ((*(ip1_enm_grad_output[tid]+x*20+_remain_y)) * (*(ip1_enm_grad_activation[tid]+x*20+_remain_y)));
sgemm_axpy(ip1_enm_grad_weights[tid][x][_remain_y], (*(ip1_enm_grad_output[tid]+x*20+_remain_y)), data_enm_output[tid], 4);
}
}
}

// weights_update for ip1_enm
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 20; y++) {
#pragma omp atomic
		sgemm_axpy(ip1_enm_weights[x][y], -0.1, ip1_enm_grad_weights[tid][x][y], 1*4);
		sgemm_zeros(ip1_enm_grad_weights[tid][x][y], 1*4);
		sgemm_zeros(ip1_enm_grad_output[tid], 1*20);
	}
}
// weights_update for ip2_enm
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 10; y++) {
#pragma omp atomic
		sgemm_axpy(ip2_enm_weights[x][y], -0.1, ip2_enm_grad_weights[tid][x][y], 1*20);
		sgemm_zeros(ip2_enm_grad_weights[tid][x][y], 1*20);
		sgemm_zeros(ip2_enm_grad_output[tid], 1*10);
	}
}
// weights_update for label_enm
for (int x = 0; x < 1; x++) {
	for (int y = 0; y < 3; y++) {
#pragma omp atomic
		sgemm_axpy(label_enm_weights[x][y], -0.1, label_enm_grad_weights[tid][x][y], 1*10);
		sgemm_zeros(label_enm_grad_weights[tid][x][y], 1*10);
		sgemm_zeros(label_enm_grad_output[tid], 1*3);
	}
}

} // end of data instances traversal
} // end of iterative traversal

// test block
vector<int> preds;
for ( int data_idx = 0 ; data_idx < test_features.size() ; data_idx = data_idx + 1 ) {
int tid = 0;

sgemm_copy (data_enm_output[tid], test_features[data_idx], 1*4);
vector<vector<int>> cur_label (1, vector<int>(3, 0));
cur_label[0][test_labels[data_idx]] = 1;
for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 20; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip1_enm_output[tid]+x*20+y)) = 0.0;
sgemm_dp((ip1_enm_output[tid]+x*20+y), ip1_enm_weights[x][y], data_enm_output[tid], 4);
(*(ip1_enm_grad_activation[tid]+x*20+y)) = (1 - pow(tanh((*(ip1_enm_output[tid]+x*20+y))), 2));
(*(ip1_enm_output[tid]+x*20+y)) = tanh((*(ip1_enm_output[tid]+x*20+y)));
}
for (int _remain_y = 18; _remain_y < 2; _remain_y += 1) {
(*(ip1_enm_output[tid]+x*20+_remain_y)) = 0.0;
sgemm_dp((ip1_enm_output[tid]+x*20+_remain_y), ip1_enm_weights[x][_remain_y], data_enm_output[tid], 4);
(*(ip1_enm_grad_activation[tid]+x*20+_remain_y)) = (1 - pow(tanh((*(ip1_enm_output[tid]+x*20+_remain_y))), 2));
(*(ip1_enm_output[tid]+x*20+_remain_y)) = tanh((*(ip1_enm_output[tid]+x*20+_remain_y)));
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 10; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(ip2_enm_output[tid]+x*10+y)) = 0.0;
sgemm_dp((ip2_enm_output[tid]+x*10+y), ip2_enm_weights[x][y], ip1_enm_output[tid], 20);
(*(ip2_enm_grad_activation[tid]+x*10+y)) = (1 - pow(tanh((*(ip2_enm_output[tid]+x*10+y))), 2));
(*(ip2_enm_output[tid]+x*10+y)) = tanh((*(ip2_enm_output[tid]+x*10+y)));
}
for (int _remain_y = 9; _remain_y < 1; _remain_y += 1) {
(*(ip2_enm_output[tid]+x*10+_remain_y)) = 0.0;
sgemm_dp((ip2_enm_output[tid]+x*10+_remain_y), ip2_enm_weights[x][_remain_y], ip1_enm_output[tid], 20);
(*(ip2_enm_grad_activation[tid]+x*10+_remain_y)) = (1 - pow(tanh((*(ip2_enm_output[tid]+x*10+_remain_y))), 2));
(*(ip2_enm_output[tid]+x*10+_remain_y)) = tanh((*(ip2_enm_output[tid]+x*10+_remain_y)));
}
}
}

for (int x = 0; x < 1; x += 1) {
for (int _tile_y = 0; _tile_y < 3; _tile_y += 3) {
for (int y = _tile_y; y < _tile_y + 3; y += 1) {
(*(label_enm_output[tid]+x*3+y)) = 0.0;
sgemm_dp((label_enm_output[tid]+x*3+y), label_enm_weights[x][y], ip2_enm_output[tid], 10);
(*(label_enm_output[tid]+x*3+y)) = exp((*(label_enm_output[tid]+x*3+y)));
}
}
}

// annotate for loss layer in testing stage
int pred = argmax (label_enm_output[tid], 1*3);
preds.push_back(pred);

}
// evaluate the accuracy performance
evaluate(preds, test_labels);

// deallocating memory for specific fields of data_enm
for (int i = 0; i < 5; i++) mkl_free(data_enm_grad_activation[i]);
for (int i = 0; i < 5; i++) mkl_free(data_enm_output[i]);
for (int i = 0; i < 5; i++) mkl_free(data_enm_grad_output[i]);
// deallocating memory for specific fields of ip1_enm
for (int i = 0; i < 5; i++) mkl_free(ip1_enm_grad_activation[i]);
for (int i = 0; i < 5; i++) mkl_free(ip1_enm_output[i]);
free_weights_mats(ip1_enm_weights);
for (int i = 0; i < 5; i++) free_weights_mats(ip1_enm_grad_weights[i]);
for (int i = 0; i < 5; i++) mkl_free(ip1_enm_grad_output[i]);
// deallocating memory for specific fields of ip2_enm
for (int i = 0; i < 5; i++) mkl_free(ip2_enm_grad_activation[i]);
for (int i = 0; i < 5; i++) mkl_free(ip2_enm_output[i]);
free_weights_mats(ip2_enm_weights);
for (int i = 0; i < 5; i++) free_weights_mats(ip2_enm_grad_weights[i]);
for (int i = 0; i < 5; i++) mkl_free(ip2_enm_grad_output[i]);
// deallocating memory for specific fields of label_enm
for (int i = 0; i < 5; i++) mkl_free(label_enm_grad_activation[i]);
for (int i = 0; i < 5; i++) mkl_free(label_enm_output[i]);
free_weights_mats(label_enm_weights);
for (int i = 0; i < 5; i++) free_weights_mats(label_enm_grad_weights[i]);
for (int i = 0; i < 5; i++) mkl_free(label_enm_grad_output[i]);

}