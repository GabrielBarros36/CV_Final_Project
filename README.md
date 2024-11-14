# Neural Nets from scratch in Python 

These are built without ML frameworks (PyTorch, Tensorflow, SKLearn).
They only use NumPy and the Python standard lib, so no GPU acceleration is available.
Both nets are trained and tested on the MNIST dataset (available here as a zip file).

### NeuralNet
A neural net with an input layer (784 neurons), one hidden layer (16 neurons), and an output layer (10 neurons).

### BiggerNet
A neural net with an input layer (784 neurons), three hidden layers (32 neurons each) and an output layer (10 neurons).

### Usage
For both nets, train by executing cell 12 and test with cell 13. You need to execute all the other cells at least once after downloading the repo, so you could use the "run all cells" button the in "run" menu. 

When training you can choose the number of epochs and the learning rate. 5 epochs with a learning rate of 0.001 gives me ~95% accuracy - it's cool to tweak the parameters and see what happens.
