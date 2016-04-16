#ifndef NEURON_H
#define NEURON_H
#include <string>
#include <iostream>

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
#endif
