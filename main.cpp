#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

unsigned char* read_mnist_labels(std::string &full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}
unsigned char** read_mnist_images(std::string &full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

void cout_2D_vector(std::vector <std::vector <double>> a) {
    for (int l = 0; l < a.size(); l++) {
        std::cout << "\nLayer: " << l << "\n\n[";
        for (int i = 0; i < a[l].size(); i++) {
            std::cout << a[l][i];
            if (i != a[l].size()-1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
}
void cout_vector(std::vector <double> a) {
    std::cout << "\n[";
    for (int i = 0; i < a.size(); i++) {
        std::cout << a[i];
        if (i != a.size()-1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
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
                for (int i = 0; i < size[L]; i++) {
                    z[L][i] = rnd_val(0.0, 1.0);
                    s[L][i] = z[L][i];
                }
            }
            else {
                for (int i = 0; i < size[L]; i++) {
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
        update();
    }
    void update() {
        for (int L = 1; L < num_layers; L++) {
            for (int i = 0; i < size[L]; i++) {
                for (int j = 0; j < size[L-1]; j++) {
                    weights[L][j][i] += learning_rate * s[L - 1][j] * deltas[L][i];
                }
                biases[L][i] += learning_rate*deltas[L][i];
            }
        }
    }
    void mini_batch_train(std::vector <double> &x, std::vector <double> &y, int batch_size, int epochs) {
        std::vector<std::vector<std::vector<double>>> batch_deltas;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < std::floor(x.size() / batch_size); batch++) {
                for (int i = batch * batch_size; i < batch_size * (batch + 1); i++) {
                    z[0][0] = x[i];
                    s[0][0] = x[i];
                    target.push_back(y[i]);
                    fwd_propagate();
                    bwd_propagate();
                    batch_deltas.push_back(deltas);
                    target.clear();
                }
                for (int L = 0; L < batch_deltas[0].size(); L++) {
                    for (int i = 0; i < batch_deltas[0][0].size(); i++) {
                        double sum = 0.0;
                        for (int bd = 0; bd < batch_deltas.size(); bd++) {
                            sum += batch_deltas[bd][L][i];
                        }
                        deltas[L][i] = sum;
                    }
                }
                batch_deltas.clear();
                update();
            }
        }
    }
    void fwd_propagate() {
        for (int L = 1; L < num_layers; L++) {
            for (int i = 0; i < size[L]; i++) {
                double out = 0;
                out += biases[L][i];

                for (int j = 0; j < size[L-1]; j++) {
                    out += weights[L][j][i]*s[L-1][j];
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
                for (int i = 0; i < size[L]; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < size[L+1]; j++) {
                        sum += weights[L+1][i][j] * deltas[L+1][j];
                    }
                    deltas[L][i] = (sum * d_sigmoid(z[L][i]));
                }
            }
        }
    }
};

void x_to_x_sqrd(network net, int train_size, int test_size, bool mini_batch, bool verbose) {
    std::vector <double> diff;
    std::vector<double> y;
    std::vector<double> x;

    int sample_size = train_size;
    if (mini_batch) {
        for (int sample = 0; sample < sample_size; sample++) {
            double val = rnd_val(0.0, 1.0);
            x.push_back(val);
            y.push_back(std::pow(val, 2));
        }
        net.mini_batch_train(x, y, 10, 10);

        for (int test = 0; test < test_size; test++) {
            double val = rnd_val(0.0, 1.0);
            net.z[0][0] = val;
            net.s[0][0] = val;
            net.target.push_back(std::pow(val, 2));
            net.fwd_propagate();

            if (verbose) {
                std::cout << "\nTarget:\n";
                cout_vector(net.target);

                std::cout << "\nOutput:\n";
                cout_vector(net.s[4]);
                std::cout << "\n";
            }
            diff.push_back(std::abs((net.s[4][0] - net.target[0]) / (net.target[0] + net.s[4][0])));
            net.target.clear();
        }
        std::cout << "Error:\t" <<std::accumulate(diff.begin(), diff.end(), 0.0) * 100 / diff.size() << "\n";
        diff.clear();
    }

    else {
        for (int sample = 0; sample < sample_size; sample++) {
            net.init_neurons();
            for (int i = 0; i < net.size[net.size.size() - 1]; i++) {
                y.push_back(std::pow(net.z[0][i], 2));
            }
            net.target = y;
            net.train();
            net.fwd_propagate();

            if (verbose) {
                std::cout << "\nTarget:\n";
                cout_vector(net.target);

                std::cout << "\nOutput:\n";
                cout_vector(net.s[4]);
                std::cout << "\n";
            }

            if (sample > sample_size-test_size) {
                diff.push_back(std::abs((net.s[4][0] - net.target[0]) / (net.target[0] + net.s[4][0])));
            }
            y.clear();
        }
        std::cout << std::accumulate(diff.begin(), diff.end(), 0.0)*100/diff.size() << "\n";
        diff.clear();
    }
}

int main() {
    network net;
    net.size = {1, 30, 30, 30, 1};
    net.init_random();
    std::vector<double> y;
    std::vector<double> x;


    x_to_x_sqrd(net, 100000, 100, true, false);
    return 0;
}