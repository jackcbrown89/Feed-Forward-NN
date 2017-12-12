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
void cout_vector(std::vector <double> a) {
    for (auto &i : a) {
        std::cout << i;
        std::cout << "\n";
    }
}
double rnd_val(double a, double b) {
    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::uniform_real_distribution<double> rng(a, b);

    double x = rng(rand_engine);
    return x;
}
double cost_function(double target, double s){
    double cost = target*std::log(s) + (1.0-target)*std::log(1.0-s);
    return cost;
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
    double gradient = 0;
    gradient = (target - s) / ( s * (1 - s) );

    return gradient;
}

class network {

public:
    std::vector <unsigned int> size = {};
    unsigned int num_layers = 0;
    double learning_rate = 0.001;
    std::vector <double> target;
    std::vector <std::vector <double>> biases;
    std::vector <std::vector <std::vector <double>>> weights;
    std::vector < std::vector <double> > deltas;
    std::vector < std::vector <double> > s;
    std::vector < std::vector <double> > z;

    void init_neurons() {
        s.resize(num_layers);
        z.resize(num_layers);
        deltas.resize(num_layers);
        for (int L = 0; L < num_layers; L++) {
            //resize all the layers
            s[L].resize(size[L]);
            deltas[L].resize(size[L]);
            z[L].resize(size[L]);
            //only fill the first layer with random input
            if (L == 0) {
                for (int i = 0; i < z[L].size(); i++) {
                    z[L][i] = rnd_val(0.0, 1.0);
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
        for (int L = 0; L < num_layers; L++) {
            if (L > 0) {
                weights[L].resize(size[L-1]);

                for (int i = 0; i < size[L-1]; i++) {
                    weights[L][i].resize(size[L]);
                    for (int j = 0; j < size[L]; j++) {
                        weights[L][i][j] = rnd_val(-1.0, 1.0);
                    }
                }
            }
            else {
                weights[L].resize(size[L]);

                for (int i = 0; i < size[L]; i++) {
                    weights[L][i].resize(size[L]);
                    for (int j = 0; j < size[L]; j++) {
                        weights[L][i][j] = 0.0;
                    }
                }
            }

        }
    }
    void init_random_biases() {
        biases.resize(num_layers);
        for (int L = 0; L < num_layers; L++) {
            biases[L].resize(size[L]);
            for (int i = 0; i < size[L]; i++) {
                if (L > 0) {
                    biases[L][i] = rnd_val(-1.0, 1.0);
                }
                else {
                    biases[L][i] = 0.0;
                }
            }
        }
    }

public: void init_random() {
        num_layers = static_cast<int> (size.size());
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
        for (auto &layer : s) {
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
        fwd_propagate();
        bwd_propagate();

        for (int L = 1; L < num_layers; L++) {
            for (int i = 0; i < size[L]; i++) {
                for (int j = 0; j < size[L-1]; j++) {
                    weights[L][j][i] += learning_rate * s[L - 1][i] * deltas[L][i];
                }
                biases[L][i] += learning_rate*deltas[L][i];
            }
        }
    }
    void fwd_propagate() {
        for (int L = 1; L < num_layers; L++) {
            for (int i = 0; i < size[L]; i++) {
                double out = 0;
                out += biases[L][i];

                for (int in_x = 0; in_x < size[L-1]; in_x++) {
                    out += weights[L][in_x][i]*s[L-1][in_x];
                }

                s[L][i] = sigmoid(out);
                z[L][i] = out;

            }
        }
    }
    void bwd_propagate() {
        for (int L = num_layers - 1; L > 0; L--) {
            if (L == num_layers - 1) {
                for (int i = 0; i < size[L]; i++) {
                    deltas[L][i] = (cost_gradient(target[i], s[L][i]) * d_sigmoid(z[L][i]));
                }
            }
            else {
                for (int i = 0; i < size[L+1]; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < size[L+1]; j++) {
                        sum += weights[L+1][j][i] * deltas[L+1][j];
                        j;
                    }
                    deltas[L][i] = (sum * d_sigmoid(z[L][i]));
                }
            }
        }
    }
};


int main() {
    network net;
    net.size = {1, 30, 1};
    net.init_random();
    std::vector<double> y;
    y.reserve(net.size[net.size.size() - 1]);

    for (int sample = 0; sample < 1; sample++) {
        net.init_neurons();
        for (int i = 0; i < net.size[net.size.size() - 1]; i++) {
            y.push_back(std::pow(net.z[0][i], 2));
        }
        net.target = y;
        auto net2 = net;


        net.train();

        // finite difference
        net2.weights[1][0][0] += .001;
        net2.fwd_propagate();
        double f1 = cost_function(net2.target[0], net2.s[2][0]);

        net2.weights[1][0][0] -= .002;
        net2.fwd_propagate();
        double f2 = cost_function(net2.target[0], net2.s[2][0]);

        double result = (f1-f2)/(.002);

        std::cout << result << "\n";
        std::cout << net.s[0][0]*net.deltas[1][0] << "\n\n";

//        std::cout << "\nTarget:\n";
//        cout_vector(net.target);
//
//        std::cout << "\nOutput:\n";
//        cout_vector(net.s[2]);

        y.clear();
    }

    return 0;
}