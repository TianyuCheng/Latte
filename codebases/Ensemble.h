#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <string>
#include <iostream>
#include <map>
#include <cassert>
#include "Neuron.h"

using namespace std;

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
#endif
