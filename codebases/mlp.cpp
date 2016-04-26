#include "Latte.h"

//using namespace std;

int main (int argn, char** argv) { 

    // create neural networks 
    Network net;

    // create ensembles used in neural networks
    Ensemble data_enm(1, 4, NULL); net.add_ensemble(&data_enm);
    Ensemble ip1_enm(1, 20, &data_enm); net.add_ensemble(&ip1_enm);
    Ensemble ip2_enm(1, 10, &ip1_enm); net.add_ensemble(&ip2_enm);
    Ensemble label_enm(1, 3, &ip2_enm); net.add_ensemble(&label_enm);

    // allocating memory for specific fields of data_enm
    vector<vector<float*>> data_enm_inputs (1, vector<float*>(4, NULL));
    float* data_enm_output = init_mkl_mat(1, 4);
    // allocating memory for specific fields of ip1_enm
    vector<vector<float*>> ip1_enm_inputs (1, vector<float*>(20, NULL));
    vector<vector<float*>> ip1_enm_grad_inputs (1, vector<float*>(20, NULL));
    vector<vector<float*>> ip1_enm_grad_weights (1, vector<float*>(20, NULL));
    float* ip1_enm_grad_activation = init_mkl_mat(1, 20);
    vector<vector<float*>> ip1_enm_weights (1, vector<float*>(20, NULL));
    float* ip1_enm_grad_output = init_mkl_mat(1, 20);
    float* ip1_enm_output = init_mkl_mat(1, 20);
    // allocating memory for specific fields of ip2_enm
    vector<vector<float*>> ip2_enm_inputs (1, vector<float*>(10, NULL));
    vector<vector<float*>> ip2_enm_grad_inputs (1, vector<float*>(10, NULL));
    vector<vector<float*>> ip2_enm_grad_weights (1, vector<float*>(10, NULL));
    float* ip2_enm_grad_activation = init_mkl_mat(1, 10);
    vector<vector<float*>> ip2_enm_weights (1, vector<float*>(10, NULL));
    float* ip2_enm_grad_output = init_mkl_mat(1, 10);
    float* ip2_enm_output = init_mkl_mat(1, 10);
    // allocating memory for specific fields of label_enm
    vector<vector<float*>> label_enm_inputs (1, vector<float*>(3, NULL));
    vector<vector<float*>> label_enm_grad_inputs (1, vector<float*>(3, NULL));
    vector<vector<float*>> label_enm_grad_weights (1, vector<float*>(3, NULL));
    vector<vector<float*>> label_enm_weights (1, vector<float*>(3, NULL));
    float* label_enm_grad_output = init_mkl_mat(1, 3);
    float* label_enm_output = init_mkl_mat(1, 3);

    // initialize weights of layers 
    init_weights_mats(ip1_enm_weights, 1, 4); 
    init_weights_mats(ip2_enm_weights, 1, 20); 
    init_weights_mats(label_enm_weights, 1, 10); 
    init_weights_mats(ip1_enm_grad_weights, 1, 4); 
    init_weights_mats(ip2_enm_grad_weights, 1, 20); 
    init_weights_mats(label_enm_grad_weights, 1, 10); 

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

        for ( int si = 0 ; si < train_features.size() ; si = si + 1 ) {

            int data_idx = shuffle_index[si];sgemm_copy (data_enm_output, train_features[data_idx], 1*4);
            vector<vector<int>> cur_label (1, vector<int>(3, 0));
            cur_label[0][train_labels[data_idx]] = 1;
            float dp_result;


            // Forward Propagation for ip1_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 20; y++) {
                    dp_result = 0.0;
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 20; ++j) {
                            dp_result = (dp_result + ((*(ip1_enm_weights[x][y]+i*20+j)) * (*(data_enm_output+i*20+j))));
                        }
                    }
                    *(ip1_enm_output+x*20+y) = tanh(dp_result);
                    *(ip1_enm_grad_activation+x*20+y) = (1 - pow(tanh(dp_result), 2));
                }
            }

            // Forward Propagation for ip2_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 10; y++) {
                    dp_result = 0.0;
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 10; ++j) {
                            dp_result = (dp_result + ((*(ip2_enm_weights[x][y]+i*10+j)) * (*(ip1_enm_output+i*10+j))));
                        }
                    }
                    *(ip2_enm_output+x*10+y) = tanh(dp_result);
                    *(ip2_enm_grad_activation+x*10+y) = (1 - pow(tanh(dp_result), 2));
                }
            }

            // Forward Propagation for label_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 3; y++) {
                    dp_result = 0.0;
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            dp_result = (dp_result + ((*(label_enm_weights[x][y]+i*3+j)) * (*(ip2_enm_output+i*3+j))));
                        }
                    }
                    *(label_enm_output+x*3+y) = exp(dp_result);
                }
            }

            // annotate for loss layer
            float sumover = 0.0;
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 3; y++) {
                    sumover += *(label_enm_output+x*3+y);
                }
            }
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 3; y++) {
                    *(label_enm_output+x*3+y) = *(label_enm_output+x*3+y) / sumover;
                }
            }

            // Backward Propagation for label_enm
            for (int x = 0; x < 1; x ++) {
                for (int y = 0; y < 3; y ++) {
                    *(label_enm_grad_output+x*3+y) = (*(label_enm_output+x*3+y) - cur_label[x][y]);
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            *(ip2_enm_output+i*3+j) = ((*(label_enm_grad_output+x*3+y)) * (*(label_enm_weights[x][y]+i*3+j)));
                        }
                    }
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            *(label_enm_grad_weights[x][y]+i*3+j) = (*(label_enm_grad_weights[x][y]+i*3+j) + ((*(label_enm_grad_output+x*3+y)) * (*(ip2_enm_output+i*3+j))));
                        }
                    }
                }
            }

            // Backward Propagation for ip2_enm
            for (int x = 0; x < 1; x ++) {
                for (int y = 0; y < 10; y ++) {
                    *(ip2_enm_grad_output+x*10+y) = ((*(ip2_enm_grad_output+x*10+y)) * (*(ip2_enm_grad_activation+x*10+y)));
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 10; ++j) {
                            *(ip1_enm_output+i*10+j) = ((*(ip2_enm_grad_output+x*10+y)) * (*(ip2_enm_weights[x][y]+i*10+j)));
                        }
                    }
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 10; ++j) {
                            *(ip2_enm_grad_weights[x][y]+i*10+j) = (*(ip2_enm_grad_weights[x][y]+i*10+j) + ((*(ip2_enm_grad_output+x*10+y)) * (*(ip1_enm_output+i*10+j))));
                        }
                    }
                }
            }

            // Backward Propagation for ip1_enm
            for (int x = 0; x < 1; x ++) {
                for (int y = 0; y < 20; y ++) {
                    *(ip1_enm_grad_output+x*20+y) = ((*(ip1_enm_grad_output+x*20+y)) * (*(ip1_enm_grad_activation+x*20+y)));
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 20; ++j) {
                            *(data_enm_output+i*20+j) = ((*(ip1_enm_grad_output+x*20+y)) * (*(ip1_enm_weights[x][y]+i*20+j)));
                        }
                    }
                    for (int i = 0; i < 1; ++i) {
                        for (int j = 0; j < 20; ++j) {
                            *(ip1_enm_grad_weights[x][y]+i*20+j) = (*(ip1_enm_grad_weights[x][y]+i*20+j) + ((*(ip1_enm_grad_output+x*20+y)) * (*(data_enm_output+i*20+j))));
                        }
                    }
                }
            }



            // weights_update for ip1_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 20; y++) {
                    sgemm_axpy(ip1_enm_weights[x][y], -0.1, ip1_enm_grad_weights[x][y], 1*4);
                    sgemm_zeros(ip1_enm_grad_weights[x][y], 1*4);
                    sgemm_zeros(ip1_enm_grad_output, 1*20);
                }
            }
            // weights_update for ip2_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 10; y++) {
                    sgemm_axpy(ip2_enm_weights[x][y], -0.1, ip2_enm_grad_weights[x][y], 1*20);
                    sgemm_zeros(ip2_enm_grad_weights[x][y], 1*20);
                    sgemm_zeros(ip2_enm_grad_output, 1*10);
                }
            }
            // weights_update for label_enm
            for (int x = 0; x < 1; x++) {
                for (int y = 0; y < 3; y++) {
                    sgemm_axpy(label_enm_weights[x][y], -0.1, label_enm_grad_weights[x][y], 1*10);
                    sgemm_zeros(label_enm_grad_weights[x][y], 1*10);
                    sgemm_zeros(label_enm_grad_output, 1*3);
                }
            }

        } // end of data instances traversal
    } // end of iterative traversal

    // test block
    vector<int> preds;
    for ( int data_idx = 0 ; data_idx < test_features.size() ; data_idx = data_idx + 1 ) {
        float dp_result;

        sgemm_copy (data_enm_output, test_features[data_idx], 1*4);
        vector<vector<int>> cur_label (1, vector<int>(3, 0));
        cur_label[0][test_labels[data_idx]] = 1;


        // Forward Propagation for ip1_enm
        for (int x = 0; x < 1; x++) {
            for (int y = 0; y < 20; y++) {
                dp_result = 0.0;
                for (int i = 0; i < 1; ++i) {
                    for (int j = 0; j < 20; ++j) {
                        dp_result = (dp_result + ((*(ip1_enm_weights[x][y]+i*20+j)) * (*(data_enm_output+i*20+j))));
                    }
                }
                *(ip1_enm_output+x*20+y) = tanh(dp_result);
                *(ip1_enm_grad_activation+x*20+y) = (1 - pow(tanh(dp_result), 2));
            }
        }

        // Forward Propagation for ip2_enm
        for (int x = 0; x < 1; x++) {
            for (int y = 0; y < 10; y++) {
                dp_result = 0.0;
                for (int i = 0; i < 1; ++i) {
                    for (int j = 0; j < 10; ++j) {
                        dp_result = (dp_result + ((*(ip2_enm_weights[x][y]+i*10+j)) * (*(ip1_enm_output+i*10+j))));
                    }
                }
                *(ip2_enm_output+x*10+y) = tanh(dp_result);
                *(ip2_enm_grad_activation+x*10+y) = (1 - pow(tanh(dp_result), 2));
            }
        }

        // Forward Propagation for label_enm
        for (int x = 0; x < 1; x++) {
            for (int y = 0; y < 3; y++) {
                dp_result = 0.0;
                for (int i = 0; i < 1; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        dp_result = (dp_result + ((*(label_enm_weights[x][y]+i*3+j)) * (*(ip2_enm_output+i*3+j))));
                    }
                }
                *(label_enm_output+x*3+y) = exp(dp_result);
            }
        }

        // annotate for loss layer in testing stage
        int pred = argmax (label_enm_output, 1*3);
        preds.push_back(pred);

    }
    // evaluate the accuracy performance
    evaluate(preds, test_labels);

    // deallocating memory for specific fields of data_enm
    free_weights_mats(data_enm_inputs);
    mkl_free(data_enm_output);
    // deallocating memory for specific fields of ip1_enm
    free_weights_mats(ip1_enm_inputs);
    free_weights_mats(ip1_enm_grad_inputs);
    free_weights_mats(ip1_enm_grad_weights);
    mkl_free(ip1_enm_grad_activation);
    free_weights_mats(ip1_enm_weights);
    mkl_free(ip1_enm_grad_output);
    mkl_free(ip1_enm_output);
    // deallocating memory for specific fields of ip2_enm
    free_weights_mats(ip2_enm_inputs);
    free_weights_mats(ip2_enm_grad_inputs);
    free_weights_mats(ip2_enm_grad_weights);
    mkl_free(ip2_enm_grad_activation);
    free_weights_mats(ip2_enm_weights);
    mkl_free(ip2_enm_grad_output);
    mkl_free(ip2_enm_output);
    // deallocating memory for specific fields of label_enm
    free_weights_mats(label_enm_inputs);
    free_weights_mats(label_enm_grad_inputs);
    free_weights_mats(label_enm_grad_weights);
    free_weights_mats(label_enm_weights);
    mkl_free(label_enm_grad_output);
    mkl_free(label_enm_output);

}
