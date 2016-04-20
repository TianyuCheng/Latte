#include "Latte.h"

//using namespace std;

int main (int argn, char** argv) { 

// create neural networks 
Network net;

// create ensembles used in neural networks
Ensemble data_enm (1, 4, NULL); net.add_ensemble(&data_enm);
Ensemble ip1_enm (20, 20, &data_enm); net.add_ensemble(&ip1_enm);
Ensemble ip2_enm (10, 10, &ip1_enm); net.add_ensemble(&ip2_enm);
Ensemble label_enm (1, 3, &ip2_enm); net.add_ensemble(&label_enm);

// allocating memory for Output, Grad_output Matrices
double* data_enm_output = init_mkl_mat (data_enm.dim_x, data_enm.dim_y);
double* ip1_enm_output = init_mkl_mat (ip1_enm.dim_x, ip1_enm.dim_y);
double* ip2_enm_output = init_mkl_mat (ip2_enm.dim_x, ip2_enm.dim_y);
double* label_enm_output = init_mkl_mat (label_enm.dim_x, label_enm.dim_y);
double* data_enm_grad_output = init_mkl_mat (data_enm.next->dim_x, data_enm.next->dim_y);
double* ip1_enm_grad_output = init_mkl_mat (ip1_enm.next->dim_x, ip1_enm.next->dim_y);
double* ip2_enm_grad_output = init_mkl_mat (ip2_enm.next->dim_x, ip2_enm.next->dim_y);

// initialize weights of layers 
vector<vector<double*>> ip1_enm_weights (ip1_enm.dim_x, vector<double*>(ip1_enm.dim_y, NULL));
init_weights_mats(ip1_enm_weights, ip1_enm.prev->dim_x, ip1_enm.prev->dim_y);
vector<vector<double*>> ip2_enm_weights (ip2_enm.dim_x, vector<double*>(ip2_enm.dim_y, NULL));
init_weights_mats(ip2_enm_weights, ip2_enm.prev->dim_x, ip2_enm.prev->dim_y);
vector<vector<double*>> label_enm_weights (label_enm.dim_x, vector<double*>(label_enm.dim_y, NULL));
init_weights_mats(label_enm_weights, label_enm.prev->dim_x, label_enm.prev->dim_y);

for (int iter = 0; iter < 10; iter ++) {
}

// deallocate weights of layers 
free_weights_mats(ip1_enm_weights);
free_weights_mats(ip2_enm_weights);
free_weights_mats(label_enm_weights);

// deallocating memory for Output, Grad_output Matrices
mkl_free(data_enm_output);
mkl_free(ip1_enm_output);
mkl_free(ip2_enm_output);
mkl_free(label_enm_output);
mkl_free(data_enm_grad_output);
mkl_free(ip1_enm_grad_output);
mkl_free(ip2_enm_grad_output);

}
