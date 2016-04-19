#include "Latte.h"

void add_connection(Network& net, Ensemble& enm1, Ensemble& enm2, Connection &connection) {

    return ;
}

void shared_variable_analsyis() {

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

        // 6. update weight
    }
    // 7. evaluate accuracy and timing performance
    // evaluate ();
    
}

int main (int argn, char** argv) {
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
