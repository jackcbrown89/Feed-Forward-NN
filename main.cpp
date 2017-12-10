#include <iostream>
#include <vector>
#include <random>
#include <cmath>

void cout_2D_vector(std::vector <std::vector <double>> a) {
    for (auto &layer : a) {
        std::cout << "\nLayer: \n\n";
        for (auto &i : layer) {
            std::cout << i;
            std::cout << "\n";
        }
    }
}
double rnd_val(double a, double b) {
    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::uniform_real_distribution<double> rng(a, b);

    double x = rng(rand_engine);
    return x;
}
double objective_function(double output, double label) {
    double O = std::pow(output-label, 2);
    return O;
}
double sigmoid(double a) {
    double f = 1/(1+exp(-1*a));

    return f;
}
double d_sigmoid (double z_l) {
    double ds;
    ds = std::exp(-1 * z_l) / std::pow((1 + std::exp(-1 * z_l)), 2);

    return ds;
}
double compute_error(std::vector <double> &target, std::vector <double> &output) {
    double error = 0.0;
    for (int i = 0; i < output.size(); i++) {
        error += std::pow((target[i]-output[i]), 2);
    }

    return error;
}
double compute_gradient(double new_in, double old_in, double label, double delta) {
    double gradient = (objective_function(new_in, label)-objective_function(old_in, label))/delta;
    return gradient;
}
double cost_gradient(double target, double s) {
    double gradient;
    gradient = (target - s) / ( s * (1 - s) );

    return gradient;
}

class network {
public: unsigned int size = 4;
    unsigned int num_layers = 3;
    double learning_rate = 0.001;
    std::vector <double> target;
    std::vector <std::vector <double>> neurons;
    std::vector <std::vector <double>> biases;
    std::vector <std::vector <std::vector <double>>> weights;
    std::vector < std::vector <double> > deltas;
    std::vector < std::vector <double> > s;
    std::vector < std::vector <double> > z;

private:
    void init_neurons() {
        neurons.resize(num_layers);
        s.resize(num_layers);
        z.resize(num_layers);
        deltas.resize(num_layers);
        for (int L = 0; L < num_layers; L++) {
            //resize all the layers
            neurons[L].resize(size);
            s[L].resize(size);
            deltas[L].resize(size);
            z[L].resize(size);
            //only fill the first layer with random input
            if (L == 0) {
                for (int i = 0; i < z[L].size(); i++) {
                    z[L][i] = rnd_val(-1.0, 1.0);
                    s[L][i] = sigmoid(z[L][i]);
                }
            }
            else {
                for (int i = 0; i < z[L].size(); i++) {
                    z[L][i] = 0;
                    s[L][i] = sigmoid(z[L][i]);
                }
            }
        }
    }
    void init_random_weights() {
        weights.resize(num_layers);
        for (int layer = 0; layer < num_layers; layer++) {
            weights[layer].resize(size);
            for (int i = 0; i < size; i++) {
                weights[layer][i].resize(size);
                for (int j = 0; j < size; j++) {
                    if (layer > 0) {
                        weights[layer][i][j] = rnd_val(-1.0, 1.0);
                    }
                    else {
                        weights[layer][i][j] = 0.0;
                    }
                }
            }
        }
    }
    void init_random_biases() {
        biases.resize(num_layers);
        for (int layer = 0; layer < num_layers; layer++) {
            biases[layer].resize(size);
            for (int i = 0; i < size; i++) {
                if (layer > 0) {
                    biases[layer][i] = rnd_val(-1.0, 1.0);
                }
                else {
                    biases[layer][i] = 0.0;
                }
            }
        }
    }

public: void init_random() {
        init_random_weights();
        init_random_biases();
        init_neurons();
    }
    void cout_weights() {
        for (auto &layer : weights) {
            std::cout << "\nLayer (Weights): \n\n";
            for (auto &i : layer) {
                for (auto &j : i) {
                    std::cout << j << ",\t";
                }
                std::cout << "\n";
            }
        }
    }
    void cout_neurons() {
        for (auto &layer : neurons) {
            std::cout << "\nLayer (Neurons): \n\n";
            for (auto &i : layer) {
                std::cout << i;
                std::cout << "\n";
            }
        }
    }
    void cout_biases() {
        for (auto &layer : biases) {
            std::cout << "\nLayer (Biases): \n\n";
            for (auto &i : layer) {
                std::cout << i;
                std::cout << "\n";
            }
        }
    }

    void train() {
        for (int run = 0; run < 1000; run++) {
            fwd_propagate();
            bwd_propagate();

            for (int L = 1; L < num_layers; L++) {
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        weights[L][j][i] += learning_rate * s[L - 1][i] * deltas[L][i];
                    }
                    biases[L][i] += deltas[L][i];
                }
            }
        }
    }
    void fwd_propagate() {
        for (int L = 1; L < num_layers; L++) {
            for (int i = 0; i < size; i++) {
                double out = 0;
                out += biases[L][i];

                for (int in_x = 0; in_x < size; in_x++) {
                    out += weights[L][in_x][i]*neurons[L-1][in_x];
                }

                neurons[L][i] = out;
                s[L][i] = sigmoid(out);
                z[L][i] = out;

            }
        }
    }
    void bwd_propagate() {
        for (int L = num_layers - 1; L > num_layers; L = L-1) {
            if (L == size - 1) {
                for (int i = 0; i < s.size(); i++) {
                    deltas[L].push_back(cost_gradient(target[i], s[L][i]) * d_sigmoid(z[L][i]));
                }
            }
            else {
                for (int i = 0; i < s.size(); i++) {
                    int sum = 0;
                    for (int j = 0; j < s.size(); j++) {
                        sum += weights[L+1][j][i] * deltas[L+1][j];
                    }
                    deltas[L].push_back(sum * d_sigmoid(z[L][i]));
                }
            }
        }
    }
};
int main() {
    network net;
    net.init_random();
    net.learning_rate = 0.001;

    std::vector <double> y;
    y.resize(net.size);
    for (int i = 0; i < net.size; i++) {
        y.push_back(i % 2);
    }
    net.target = y;

    std::cout << "\nZ:\n";
    cout_2D_vector(net.z);

    std::cout << "\nS:\n";
    cout_2D_vector(net.s);

    net.cout_weights();
    net.train();

    std::cout << "\nZ:\n";
    cout_2D_vector(net.z);


    return 0;
}