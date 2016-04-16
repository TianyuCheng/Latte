#include "Network.h"
#include "Ensemble.h"
#include "Neuron.h"

void add_connection (Network& net, Ensemble& enm1, Ensemble& enm2, map<int, vector<int>>& mappings) {

    return ;
}

void shared_variable_analsyis () {

    return ;
}

void Xaiver_initialize() {

    return ;
}


int ITERATIONS = 1000;
void solve (Network& network, Solver& solver) {
    // 1. mapping to shared memory region
    // INPUT: network architecture
    // OUTPUT: compact memory allocation of computing neurons 
    shared_variable_analsyis(); 
    // 2. initialize the connection (weights) parameter of network 
    Xaiver_initialize();
    // 3. 
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // 4. forward routine through pattern matching

        // 5. backward routine through pattern matching 
        // (solver like sgd is used here..)

    }
    // 6. evaluate accuracy and timing performance
    evaluate ();
    
}
