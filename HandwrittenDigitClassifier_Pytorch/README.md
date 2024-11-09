# Handwritten Digit Classifier with PyTorch

Source: [Handwritten Digit MNIST with PyTorch](https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627)

- Increasing the number of neurons (not layers) does not affect the accuracy in a meaningful way.

## Pesado code

1. Import necessary libraries
2. Defines data transforms
3. Downloads and loads training and test data
4. Defines the neural network architecture
5. Instantiates the model
6. Defines loss function and optimizer
7. Trains the model for 15 epochs
8. Tests the model on the test set and prints accuracy

The neural network has 3 fully connected layers:

1. Input layer: 784 nodes (28x28 pixels flattened)
2. Hidden layer 1: 128 nodes
3. Hidden layer 2: 64 nodes
4. Output layer: 10 nodes (one for each digit)

ReLU activation is used for hidden layers and log softmax for the output layer.
