#ifndef LATTE_H
#define LATTE_H
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cassert>

using namespace std;

class Neuron {
    public:

    static int neuron_id_counter;
    Neuron () {
        neuron_id = neuron_id_counter++;
    }

    int get_id () {
        return neuron_id;
    }

    string get_string () {
        string result = "";
        result += "Neuron ID: " + to_string(neuron_id);
        return result;
    }

    private:

    int neuron_id;
};

int Neuron::neuron_id_counter = 0;
#

class Ensemble {
    //######################################################
    public:
    //######################################################
    static int ensemble_id_counter;
    Ensemble () {
        ensemble_id = ensemble_id_counter++;
    }

    int get_id () { return ensemble_id; }
    Neuron* get_neuron (int neuron_id) {
        if (neurons.find(neuron_id) == neurons.end()) 
            assert (false && "No corresponding neuron found in this ensemble.");
        else return neurons[neuron_id];
    }

    string get_string () {
        string result = "";
        result += "Ensemble ID: " + to_string(ensemble_id);
        return result;
    }

    //######################################################
    private:
    //######################################################
    int ensemble_id;
    map<int, Neuron*> neurons;
};

int Ensemble::ensemble_id_counter = 0;

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

class Solver {
    public:
        string solver_name;
};

int Network::network_id_counter = 0;
#endif
