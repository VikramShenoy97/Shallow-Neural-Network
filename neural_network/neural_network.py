import numpy as np

class NeuralNetwork():

    def __init__(self, n_h=4, learning_rate=1.2, num_iterations=10000, verbose=False):
        self.n_h = n_h
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose

    def fit(self, X, Y):
        return self._fit(X, Y)

    def predict(self, X):
        return self._predictions(X)

    def evaluate_loss(self):
        return self._evaluate_loss()

    def accuracy(self, predictions, Y):
        return self._accuracy(predictions, Y)

    def _sigmoid(self, z):
        s = 1. / (1 + np.exp(-z))
        return s

    def _layer_sizes(self, X, Y):
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
        return (n_x, n_h, n_y)

    def _initialize_parameters(self, n_x, n_y):
        np.random.seed(2)
        W1 = np.random.randn(self.n_h, n_x)*0.01
        b1 = np.zeros((self.n_h, 1))
        W2 = np.random.rand(n_y, self.n_h)*0.01
        b2 = np.zeros((n_y, 1))

        assert(W1.shape == (self.n_h, n_x))
        assert(b1.shape == (self.n_h, 1))
        assert(W2.shape == (n_y, self.n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

        return parameters

    def _forward_propogation(self, X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self._sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2, cache

    def _compute_cost(self, A2, Y):
        m = Y.shape[1]

        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
        cost = -np.sum(logprobs) / m

        cost = np.squeeze(cost)
        assert(cost.shape == () and isinstance(cost, float))

        return cost

    def _backward_propogation(self, cache, X, Y, parameters):
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = (1. / m) * np.dot(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1,2)))
        dW1 = (1. / m) * np.dot(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dW2":dW2, "db2":db2, "dW1":dW1, "db1":db1}

        return gradients

    def _update_parameters(self, gradients, parameters):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = gradients["dW1"]
        db1 = gradients["db1"]
        dW2 = gradients["dW2"]
        db2 = gradients["db2"]

        W1 = W1 - self.learning_rate*dW1
        W2 = W2 - self.learning_rate*dW2
        b1 = b1 - self.learning_rate*db1
        b2 = b2 - self.learning_rate*db2

        parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
        return parameters

    def _fit(self, X, Y):
        np.random.seed(3)
        n_x = self._layer_sizes(X, Y)[0]
        n_y = self._layer_sizes(X,Y)[2]
        costs = []
        parameters = self._initialize_parameters(n_x,n_y)

        for i in range(0, self.num_iterations):
            A2, cache = self._forward_propogation(X, parameters)
            cost = self._compute_cost(A2, Y)
            gradients = self._backward_propogation(cache, X, Y, parameters)
            parameters = self._update_parameters(gradients, parameters)
            if(self.verbose and i%(self.num_iterations/10) == 0):
                costs.append(cost)
                print "The cost after iteration %d is: %f" %(i,cost)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        self.costs_ = costs
        self.parameters_ = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

    def _predictions(self, X):
        try:
            self.parameters_
        except AttributeError:
            raise ValueError('fit(X, Y) needs to be called before using predict(X).')

        A2, cache = self._forward_propogation(X, self.parameters_)
        predictions = np.round(A2)
        return predictions

    def _accuracy(self, predictions, Y):
        accuracy = (100 - np.mean(np.abs(predictions - Y))*100)
        return accuracy

    def _evaluate_loss(self):
        try:
            self.parameters_
        except:
            raise ValueError('fit(X, Y) needs to be called before using evaluate_loss().')
        return self.costs_
