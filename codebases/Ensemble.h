#ifndef STRING_H
#include <string>
#endif
#include <iostream>

using namespace std;

class Ensemble {
    public:

    static int ensemble_id_counter;
    Ensemble () {
        ensemble_id = ensemble_id_counter++;
    }

    int get_id () {
        return ensemble_id;
    }

    string get_string () {
        string result = "";
        result += "Ensemble ID: " + to_string(ensemble_id);
        return result;
    }

    private:

    int ensemble_id;
};

int Ensemble::ensemble_id_counter = 0;
