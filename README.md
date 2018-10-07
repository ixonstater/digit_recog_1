network.py is a simple neural network class that trains via single sample mini batch stochastic gradient descent.
	-the network class requires either a local or a global instance of numpy, the popular python data processing library
	-the class is initialized with three parameters to the constructor
		1) a python list containing the number of nodes in the first, second and third layers eg. [100, 20, 4]
		2) a floating point learning rate
		3) a floating point momentum rant (set to 0 to negate the effects of the momentum term)
	-the classes member functions should be fairly self explanatory, however it is worth mentioning that the backprop() function does not loop until an error limit is reached,
 	this is instead done by the calling program as the error term is the return value of the backprop() function.

digit_recog.py is a simple class that contains methods for the initialization of training data and simple demostrations of the neural networks potential.
