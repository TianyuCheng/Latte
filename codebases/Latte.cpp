#include "Latte.h"

int main(int argc, char *argv[])
{
    vector<float*> features;
    vector<int> labels;
    int stride = 28;
    read_mnist("../datasets/mnist-train.csv", features, labels, stride);
    return 0;
}
