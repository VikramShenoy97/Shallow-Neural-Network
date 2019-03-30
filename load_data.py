import numpy as np
import scipy.io as sio

def shuffle_in_unison(a,b):
    p = np.random.permutation(len(a))
    return a[p],b[p]

def load_dataset():
    train_input_test = sio.loadmat('datasets/input_breast.mat')
    train_output_d_test= sio.loadmat('datasets/output_breast.mat')
    train_input_test = train_input_test['input']
    train_output_d_test = train_output_d_test['output']
    train_input_test,train_output_d_test = shuffle_in_unison(train_input_test,train_output_d_test)
    samples = len(train_input_test)
    number_of_training_samples = int(samples * 0.80)
    train_input = train_input_test[:number_of_training_samples,:]
    train_output_d = train_output_d_test[:number_of_training_samples]
    test_input = train_input_test[number_of_training_samples:,:]
    test_output_d = train_output_d_test[number_of_training_samples:]

    return train_input.T, train_output_d.T, test_input.T, test_output_d.T
