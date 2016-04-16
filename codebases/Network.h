#ifndef NETWORK_H
#define NETWORK_H
#include <string>
#include <iostream>
#include <map>
#include <cassert>
#include "Ensemble.h"
#include "Neuron.h"

using namespace std;

class Network {
    //######################################################
    public:
    //######################################################
    static int network_id_counter;
    Network () {
        network_id = network_id_counter++;
    }

    int get_id () { return network_id; }

    Ensemble* get_ensemble (int enm_id) {
        if (ensembles.find(enm_id) == ensembles.end()) 
            assert(false && "No ensemble found in this network. ");
        else return ensembles[enm_id];
    }

    Neuron* get_neuron (int neuron_id) {
        if (neurons.find(neuron_id) == neurons.end()) 
            assert(false && "No neuron found in this network. ");
        else return neurons[neuron_id];
    }

    string get_string () {
        string result = "";
        result += "Network ID: " + to_string(network_id);
        return result;
    }

    //######################################################
    private:
    //######################################################
    int network_id;
    map<int, Ensemble*> ensembles;
    map<int, Neuron*> neurons;
};

int Network::network_id_counter = 0;
#endif
