from three_layer_neural_network import *

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class is a deep neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dims, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        Constructs a deep neural network
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param nn_hidden_dims: an array of length equal to the # of layers, values equal to # of neuron
        :param actFun_type: type of activation function to be used
        :param reg_lambda: regularization parameter
        :param seed: seed of the rng
        """

        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.nn_hidden_dims = nn_hidden_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.n_layers = len(nn_hidden_dims)

        # Initialize the network
        np.random.seed(seed)
        self.hidden_layers = []
        prev_dim = nn_input_dim
        for n in nn_hidden_dims:
            l = Layer(n, prev_dim, self.actFun_type, seed)
            self.hidden_layers.append(l)
            prev_dim = n

    def feedforward(self, X):
        """
        Given an input, calculates the output of the network
        :param X: the input
        :return: none, modifies self.probs
        """
        activation_function =  lambda x: self.actFun(x, type=self.actFun_type)
        self.hidden_layers[0].feedforward(X, activation_function)
        # do the rest of the layers
        # do the output layer
        return None
    def backprop(self, X, y):
        return None
    def calculate_loss(self, X, y):
        return None
    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        return None

class Layer(object):
    """
    This is one hidden layer in the n-layer network
    """
    def __init__(self, nn_dim, nn_prev_dim, actFun_type='tanh', seed=0):
        """
        :param nn_dim: number of hidden neurons
        :param nn_prev_dim: number of hidden neurons in the previous layer
        :param actFun_type: type of activation function to be used
        :param seed: value for the random seed during initialization
        """
        self.nn_dim = nn_dim
        self.nn_prev_dim = nn_prev_dim
        self.actFun_type = actFun_type

        # Initialize weight and bias
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_prev_dim, self.nn_dim) / np.sqrt(self.nn_prev_dim)
        self.b = np.zeros((1, self.nn_dim))

    def feedforward(self, a, actFun):
        """
        :param a: activation from previous layer
        :param actFun: the activation function passed as an anonymous function
        :return: None
        """
        self.z = a * self.W + self.b
        self.a = actFun(self.z)
        return None

    def backprop(self, a, z, delta_next, diff_actFun):
        """
        :param a: the activation from the previous layer
        :param z: the z from the previous layer
        :param delta_next: the delta from the next layer
        :param diff_actFun: the differentiated activation function as an anonymous function
        :return: gradient of weight and bias for the layer
        """
        self.delta = diff_actFun(z) * np.dot(delta_next, self.W.transpose())
        self.db = np.sum(delta_next, axis=0)
        self.dW = np.dot(self.a.transpose(), delta_next)
        return self.dW, self.db

def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dims=[3, 2], nn_output_dim=2, actFun_type='relu')
    print model.hidden_layers[1].W
    # model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()