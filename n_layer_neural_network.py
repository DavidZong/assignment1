from three_layer_neural_network import *

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class is a deep neural network
    """
    def __init__(self):


    def feedforward(self, X, actFun):

    def backprop(self, X, y):

    def calculate_loss(self, X, y):

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):


class Layer(object):
    """
    This is one hidden layer in the n-layer network
    """
    def __init__(self, nn_dim, nn_prev_dim, actFun_type='tanh', seed=0):
        """
        :param nn_dim: number of hidden neurons
        :param nn_prev_dim: number of hidden neurons in the previous layer
        :param actFun_type: type of activation function to be used
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
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=50, nn_output_dim=2, actFun_type='relu')
    # model.fit_model(X,y)
    # model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()