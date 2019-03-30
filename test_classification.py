import numpy as np

from load_data import load_dataset
from draw_graph import drawGraph
from neural_network import NeuralNetwork


np.random.seed(1)
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

number_of_epochs = 10000
NN = NeuralNetwork(n_h=6, learning_rate=1.2, num_iterations=number_of_epochs, verbose=True)
NN.fit(train_set_x, train_set_y)
training_loss = NN.evaluate_loss()
print "Training Loss = %f" %(training_loss[-1])
predictions = NN.predict(train_set_x)
training_accuracy = NN.accuracy(predictions, train_set_y)
print "Training Accuracy = %f" %(training_accuracy) +"%"
predictions = NN.predict(test_set_x)
testing_accuracy = NN.accuracy(predictions, test_set_y)
print "Accuracy = %f" %(testing_accuracy) +"%"
drawGraph(number_of_epochs, training_loss, training_accuracy, testing_accuracy)
