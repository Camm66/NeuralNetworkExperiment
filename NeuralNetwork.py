import numpy
import scipy.special

def networkSetup():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


#neural network class definition
class neuralNetwork:

    #initialise the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #Set the nodes for each layer
        self.inputNodes = inputnodes
        self.hiddenNodes = hiddennodes
        self.outputNodes = outputnodes
        
        #Set the learning rate
        self.lr = learningrate
        
        #Set the link weights for our nodes
        #IE) as a matrix sampled from a normal distribution function
        self.weightsInputToHidden = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.weightsHiddenToOutput = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        
        #Set the activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    #train the network
    def train(self, inputs_list, targets_list):
        #Convert input values list into a 2d Array
        inputs = numpy.array(inputs_list, ndmin = 2).T
        #Convert target values list into a 2d Array
        targets = numpy.array(targets_list, ndmin = 2).T
        
        #Calculate signals going into the Hidden Layer
        hidden_inputs = numpy.dot(self.weightsInputToHidden, inputs)
        #Calculate signals emerging from the Hidden Layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Calculate signals going into the Output Layer
        final_inputs = numpy.dot(self.weightsHiddenToOutput, hidden_outputs)
        #Calculate signals emerging from the Output Layer
        final_outputs = self.activation_function(final_inputs)        
        
        #Output Layer error is (target - actual)
        output_errors = targets - final_outputs
        
        #Hidden Layer error is output_errors, split by weights
        #recombined at hidden nodes
        hidden_errors = numpy.dot(self.weightsHiddenToOutput.T, output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.weightsHiddenToOutput += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                                          numpy.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layers
        self.weightsInputToHidden += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                         numpy.transpose(inputs))
        pass

    #query the network
    def query(self, inputs_list):
        #Convert input list into a 2d Array
        inputs = numpy.array(input_list, ndmin = 2).T
        
        #Calculate signals going into the Hidden Layer
        hidden_inputs = numpy.dot(self.weightsInputToHidden, inputs)
        #Calculate signals emerging from the Hidden Layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Calculate signals going into the Output Layer
        final_inputs = numpy.dot(self.weightsHiddenToOutput, hidden_outputs)
        #Calculate signals emerging from the Output Layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
