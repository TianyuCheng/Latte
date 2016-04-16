#ifndef STRING_H
#include <string>
#endif
#include <iostream>

using namespace std;

class Network {
    public:

    static int network_id_counter;
    Network () {
        network_id = network_id_counter++;
    }

    int get_id () {
        return network_id;
    }

    string get_string () {
        string result = "";
        result += "Network ID: " + to_string(network_id);
        return result;
    }

    private:

    int network_id;
};

int Network::network_id_counter = 0;
