

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

        #set the nodes for each layer
        self.inputNodes = inputnodes
        self.hiddenNodes = hiddennodes
        self.outputNodes = outputnodes

        #learning rate
        self.lr = learningrate

    #train the network
    def train():

    #query the network
    def query():
