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

        # Make the hidden layers
        self.hidden_layers = []
        prev_dim = nn_input_dim
        for n in nn_hidden_dims:
            l = Layer(n, prev_dim, self.actFun_type)
            self.hidden_layers.append(l)
            prev_dim = n

        # Make the output layer
        self.oW = np.random.randn(self.nn_hidden_dims[-1], self.nn_output_dim) / np.sqrt(self.nn_hidden_dims[-1])
        self.ob = np.zeros((1, self.nn_output_dim))
    def feedforward(self, X, activation_function):
        """
        Given an input, calculates the output of the network
        :param X: the input
        :return: none, modifies self.probs
        """
        # activation_function =  lambda x: self.actFun(x, type=self.actFun_type)
        last_a = X
        for layer in self.hidden_layers:
            layer.feedforward(last_a, activation_function)
            last_a = layer.a

        # do the output layer
        zout = np.dot(last_a, self.oW) + self.ob
        exp_scores = np.exp(zout)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def backprop(self, X, y):
        """
        Performs backpropagation on the network, finding all the dW and db
        :param X: input data
        :param y: labels
        :return: dW and db are arrays of np arrays that are the changes
        """
        num_examples = len(X)
        delta_last = self.probs
        delta_last[range(num_examples), y] -= 1
        diff_actFun = lambda x: self.diff_actFun(x, type=self.actFun_type)

        # backprop the output layer
        a = self.hidden_layers[-1].a
        z = self.hidden_layers[-1].z
        odb = np.sum(delta_last, axis=0)
        odW = np.dot(a.transpose(), delta_last)
        delta_output = diff_actFun(z) * np.dot(delta_last, self.oW.transpose())
        dW = [odW]
        db = [odb]

        # backprop the hidden layer(s)
        delta_next = delta_output
        l = self.n_layers - 1
        while l >= 0:
            if l == 0:
                a = X
                z = X # this isn't needed on the 1st layer, but is here since it has the correct dimensions
            else:
                a = self.hidden_layers[l-1].a
                z = self.hidden_layers[l-1].z
            layer_dW, layer_db = self.hidden_layers[l].backprop(a, z, delta_next, diff_actFun)
            delta_next = self.hidden_layers[l].delta
            dW.insert(0, layer_dW)
            db.insert(0, layer_db)
            l -= 1
        return dW, db

    def calculate_loss(self, X, y):
        loss = 1
        return loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            # Feedforward
            activation_function = lambda x: self.actFun(x, type=self.actFun_type)
            self.feedforward(X, activation_function)

            # Backprop
            dW, db = self.backprop(X, y)


            # Apply Regularization

            # Update
            # output layer
            self.oW += -epsilon*dW[-1]
            self.ob += -epsilon*db[-1]

            for l in range(0, self.n_layers):
                layer = self.hidden_layers[l]
                layer.W += -epsilon * dW[l]
                layer.b += -epsilon * db[l]

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
        return None

class Layer(object):
    """
    This is one hidden layer in the n-layer network
    """
    def __init__(self, nn_dim, nn_prev_dim, actFun_type='tanh'):
        """
        :param nn_dim: number of hidden neurons
        :param nn_prev_dim: number of hidden neurons in the previous layer
        :param actFun_type: type of activation function to be used
        """
        self.nn_dim = nn_dim
        self.nn_prev_dim = nn_prev_dim
        self.actFun_type = actFun_type

        # Initialize weight and bias
        self.W = np.random.randn(self.nn_prev_dim, self.nn_dim) / np.sqrt(self.nn_prev_dim)
        self.b = np.zeros((1, self.nn_dim))

    def feedforward(self, a, actFun):
        """
        :param a: activation from previous layer
        :param actFun: the activation function passed as an anonymous function
        :return: None
        """
        self.z = np.dot(a, self.W) + self.b
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
        self.dW = np.dot(a.transpose(), delta_next)
        return self.dW, self.db

def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()


    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dims=[3,3,3], nn_output_dim=2, actFun_type='relu')
    # activation_function = lambda x: model.actFun(x, type=model.actFun_type)
    # model.feedforward(X, activation_function)
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)
    print model.probs


if __name__ == "__main__":
    main()