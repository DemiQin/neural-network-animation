import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from math import fabs
from formulae import sigmoid, sigmoid_derivative, random_weight, get_synapse_colour, adjust_line_to_perimeter_of_circle, layer_left_margin, get_circle_colour
import parameters


class Synapse():
    def __init__(self, input_neuron_index, x1, x2, y1, y2):
        self.input_neuron_index = input_neuron_index
        self.weight = random_weight()
        self.signal = 0
        x1, x2, y1, y2 = adjust_line_to_perimeter_of_circle(x1, x2, y1, y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def set_weight(self, input_neuron_index, x1, x2, y1, y2, weight):
        self.input_neuron_index = input_neuron_index
        self.signal = 0
        x1, x2, y1, y2 = adjust_line_to_perimeter_of_circle(x1, x2, y1, y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.weight = weight

    def draw(self):
        # print self.x1, self.x2, self.y1, self.y2, self.weight, get_synapse_colour(self.weight)
        line = pyplot.Line2D((self.x1, self.x2), (self.y1, self.y2), lw=fabs(self.weight), color=get_synapse_colour(self.weight), zorder=1)
        # let the transparency be
        outer_glow = pyplot.Line2D((self.x1, self.x2), (self.y1, self.y2), lw=(fabs(self.weight) * 2),
                                   color=get_synapse_colour(self.weight), zorder=2, alpha=(round(2.0 * math.fabs(1.0/(1.0 + math.exp(self.weight * -1.0)) - 0.5), 3)))
        pyplot.gca().add_line(line)
        pyplot.gca().add_line(outer_glow)


class Neuron():
    def __init__(self, x, y, previous_layer):
        self.x = x
        self.y = y
        self.output = 0
        self.synapses = []
        self.error = 0
        index = 0
        if previous_layer:
            for input_neuron in previous_layer.neurons:
                synapse = Synapse(index, x, input_neuron.x, y, input_neuron.y)
                self.synapses.append(synapse)
                index += 1

    def add_weight(self, x, y, previous_layer, weight):# previous_layer to this neuron
        self.x = x
        self.y = y
        self.output = 0
        self.synapses = []
        self.error = 0
        index = 0
        if previous_layer:
            for i in range(len(previous_layer.neurons)):
                input_neuron = previous_layer.neurons[i]
                synapse = Synapse(index, x, input_neuron.x, y, input_neuron.y)
                # print i, len(weight)
                synapse.set_weight(index, x, input_neuron.x, y, input_neuron.y, weight[i])
                self.synapses.append(synapse)
                index += 1




    def train(self, previous_layer):
        for synapse in self.synapses:
            # Propagate the error back down the synapse to the neuron in the layer below
            previous_layer.neurons[synapse.input_neuron_index].error += self.error * sigmoid_derivative(self.output) * synapse.weight
            # Adjust the synapse weight
            synapse.weight += synapse.signal * self.error * sigmoid_derivative(self.output)
        return previous_layer

    def think(self, previous_layer):
        activity = 0
        for synapse in self.synapses:
            synapse.signal = previous_layer.neurons[synapse.input_neuron_index].output
            activity += synapse.weight * synapse.signal
        self.output = sigmoid(activity)

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=parameters.neuron_radius, fill=True, color=(0.2, 0.2, 0), zorder=3)# zorder is drawing order, big means front
        # this one used as highlight the input in original project
        try:
            alpha_pre = (1.0 / ( 1.0 + math.exp(-1.0 * self.output))) * 0.5
        except:
            alpha_pre = 0.5
        outer_glow = pyplot.Circle((self.x, self.y), radius=parameters.neuron_radius * 1.5, fill=True,
                                   color=get_circle_colour(self.output), zorder=4, alpha=alpha_pre)# alpha: tranparency
        pyplot.gca().add_patch(circle)
        pyplot.gca().add_patch(outer_glow)
        pyplot.text(self.x, self.y - 1, round(self.output, 1), fontsize=5)
        for synapse in self.synapses:
            synapse.draw()


class Layer():
    # def __init__(self, network, number_of_neurons):
    #     if len(network.layers) > 0:
    #         self.is_input_layer = False
    #         self.previous_layer = network.layers[-1]
    #         self.y = self.previous_layer.y + parameters.vertical_distance_between_layers
    #     else:
    #         self.is_input_layer = True
    #         self.previous_layer = None
    #         self.y = parameters.bottom_margin
    #     self.neurons = []
    #     x = layer_left_margin(number_of_neurons)
    #     for iteration in xrange(number_of_neurons):
    #         neuron = Neuron(x, self.y, self.previous_layer)
    #         self.neurons.append(neuron)
    #         x += parameters.horizontal_distance_between_neurons

    # why the call later don't need to appoint the network?
    def __init__(self, network, number_of_neurons, weight):# weight: previous layer to this layer
        if len(network.layers) > 0:
            self.is_input_layer = False
            self.previous_layer = network.layers[-1]
            self.y = self.previous_layer.y + parameters.vertical_distance_between_layers
        else:
            self.is_input_layer = True
            self.previous_layer = None
            self.y = parameters.bottom_margin
        self.neurons = []
        x = layer_left_margin(number_of_neurons)
        for i in range(number_of_neurons):
            neuron = Neuron(x, self.y, self.previous_layer)
            if len(weight) > 1:
                neuron.add_weight(x, self.y, self.previous_layer, weight[i])# error!!!!!!!!!!!!!!, this one need no return!!!!!!!!!!!!
            self.neurons.append( neuron)
            x += parameters.horizontal_distance_between_neurons


    def think(self):
        for neuron in self.neurons:
            neuron.think(self.previous_layer)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()

import codecs
class NeuralNetwork():

    def __init__(self, requested_layers):
        self.layers = []
        self.invmead = []
        self.invstd = []
        for number_of_neurons in requested_layers:
            self.layers.append(Layer(self, number_of_neurons, []))

    def load(self, model_file):
        #the format is changeable,so check it out when use
        fr = codecs.open(model_file, 'r')
        line = fr.readline()
        btokens = ['B0', 'B1']
        wtokens = ['W0', 'W1']
        self.invstd = []
        self.invmead = []
        tokens_dic = {}
        while line:
            for i in range(len(btokens)):
                if line.startswith(btokens[i]) and not line.__contains__("*"):
                    tokens_dic[btokens[i]] = []
                    line = fr.readline()
                    while line and not line.__contains__('##'):
                        tokens_dic[btokens[i]].append(float(line.strip()))
                        line = fr.readline()
                if line.startswith(wtokens[i]) and not line.__contains__("*"):
                    tokens_dic[wtokens[i]] = []
                    line = fr.readline()
                    while line and not line.__contains__('##'):
                        tokens_dic[wtokens[i]].append([float(x) for x in line.strip().split()])
                        line = fr.readline()
                if line.startswith("InvStdOfFeatures"):
                    line = fr.readline()
                    while line and not line.__contains__('##'):
                        self.invstd.append(float(line.strip()))
                        line = fr.readline()
                if line.startswith("MeanOfFeatures"):
                    line = fr.readline()
                    while line and not line.__contains__('##'):
                        self.invmead.append(float(line.strip()))
                        line = fr.readline()
            line = fr.readline()#skip divider line
        # check the weight is readable and correct
        # H1=Sigmoid ( W0*features+B0 ) ; HLAST = W1*H0+B1
        self.layers =[]# this also cannot make the network
        self.layers.append(Layer(self, len(tokens_dic['W0'][0]), []))# use parameter
        self.layers.append(Layer(self, len(tokens_dic['W1'][0]), tokens_dic['W0']))
        # print "layer3 start"
        self.layers.append(Layer(self, len(tokens_dic['B1']), tokens_dic['W1']))

    def train(self, example):
        error = example.output - self.think(example.inputs)
        self.reset_errors()
        self.layers[-1].neurons[0].error = error
        for l in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[l].neurons:
                self.layers[l - 1] = neuron.train(self.layers[l - 1])
        return fabs(error)

    def do_not_think(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.output = 0
                for synapse in neuron.synapses:
                    synapse.signal = 0
    # restore the calculate process to get the exact output
    def think(self, inputs):
        for layer in self.layers:
            if layer.is_input_layer:
                for index, value in enumerate(inputs):
                    self.layers[0].neurons[index].output = (value - self.invmead[index])/(0.00001 + math.sqrt(self.invstd[index]))
            else:
                layer.think()
        return self.layers[-1].neurons[0].output
    # def think(self, inputs):
    #     for layer in self.layers:
    #         if layer.is_input_layer:
    #             for index, value in enumerate(inputs):
    #                 self.layers[0].neurons[index].output = value
    #         else:
    #             layer.think()
    #     return self.layers[-1].neurons[0].output

    def draw(self):
        pyplot.cla()
        for layer in self.layers:
            layer.draw()

    def reset_errors(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.error = 0