import numpy

def networkSetup():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    numpy.random.rand(3, 3) -0.5


#neural network class definition
class neuralNetwork:

    #initialise the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #set the nodes for each layer
        self.inputNodes = inputnodes
        self.hiddenNodes = hiddennodes
        self.outputNodes = outputnodes
        
        #set the link weights for our nodes
        #IE) as a matrix sampled from a normal distribution function
        self.weightsInputToHidden = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.weightsaHiddenToOutput = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        #learning rate
        self.lr = learningrate
        pass

    #train the network
    def train():
        pass

    #query the network
    def query():
        pass
