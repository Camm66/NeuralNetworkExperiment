import numpy
import scipy.special
import scipy.ndimage
import imageio
import matplotlib.pyplot
#%matplotlib inline

def networkSetup():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    trainingData(n, output_nodes)
    testingData(n)
    testImage(n)
    pass

def trainingData(n, output_nodes):
    data_file = open("TrainingData/mnist_train.csv", 'r')
    training_data = []
    line = data_file.readline()
    while line:
        training_data.append(line)
        line = data_file.readline()
    data_file.close()

    #Number of times the training data is run through the network
    epochs = 5

    for e in range(epochs):
        for record in training_data:
            #Parse the input data, Scale it for use with the network
            all_values = record.split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            #Set the target array for the node
            #IE) [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] == 3
            target_values = numpy.zeros(output_nodes) + 0.01
            target_values[int(all_values[0])] = 0.99

            n.train(scaled_input, target_values)

            #Train inputs rotated +/- 10 degrees!
            input_plus10deg = scipy.ndimage.interpolation.rotate(
                scaled_input.reshape(28,28), 10, cval = 0.01, reshape = False)
            n.train(input_plus10deg.reshape(784), target_values)

            input_minus10deg = scipy.ndimage.interpolation.rotate(
                scaled_input.reshape(28,28), -10, cval = 0.01, reshape = False)
            n.train(input_minus10deg.reshape(784), target_values)
            pass
        pass


def testingData(n):
    data_file = open("TrainingData/mnist_test.csv", 'r')
    testing_data = []
    line = data_file.readline()
    while line:
        testing_data.append(line)
        line = data_file.readline()
    data_file.close()

    scoreboard = []

    for record in testing_data:
        all_values = record.split(',')
        correct_answer = int(all_values[0])
        #print(correct_answer, "correct answer")
        scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        output_values = n.query(scaled_input)
        network_answer = numpy.argmax(output_values)
        #print(network_answer, "network's answer")

        if (network_answer == correct_answer):
            scoreboard.append(1)
        else:
            scoreboard.append(0)


    scoreboard_array = numpy.asarray(scoreboard)
    successRate = (scoreboard_array.sum()/ float(scoreboard_array.size)) * 100
    print("Success rate: %s" % successRate)
    pass


def testImage(n):
    image_num = 3

    img_array = imageio.imread("2828_my_own_3.png", as_gray=True)
    img_data = 255.0 - img_array.reshape(n.inputNodes)
    scaled_img_data = (img_data / 255.0 * 0.99) + 0.01

    output_value = n.query(scaled_img_data)
    network_answer = numpy.argmax(output_value)

    matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

    print("Queried image: %s" % image_num)
    print("Network answer: %s" % network_answer)
    if image_num == network_answer:
        print("Correct number Identified!")


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
    def train(self, input_list, target_list):
        #Convert input values list into a 2d Array
        inputs = numpy.array(input_list, ndmin = 2).T
        #Convert target values list into a 2d Array
        targets = numpy.array(target_list, ndmin = 2).T

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
    def query(self, input_list):
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

if __name__ == '__main__':
    networkSetup()
