from neural_network import NeuralNetwork
from formulae import calculate_average_error, seed_random_number_generator
from video import generate_writer, annotate_frame, take_still
import parameters


class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

if __name__ == "__main__":

    # Seed the random number generator
    seed_random_number_generator()

    # Assemble a neural network, with 18 neurons in the first layer
    # 32 neurons in the second layer and 2 neuron in the third layer
    network = NeuralNetwork([18, 32, 2])

    # Training set
    examples = [TrainingExample([-29.35549,-25.04468,-63.07045,-70.7681,11,10,10,2,-122.4435,-25.93066,-25.31638,-89.88123,-87.62306,6,13,11,0,-112.9629],1.0),
                TrainingExample([-25.93066,-25.31638,-89.88123,-87.62306,6,13,11,0,-112.9629,-29.35549,-25.04468,-63.07045,-70.7681,11,10,10,2,-122.4435],0.0),
                TrainingExample([-81.19243,-101.9763,-167.3187,-200.8598,47,23,31,10,-179.9829,-91.31429,-114.3718,-170.2887,-209.5479,51,24,32,10,-178.6222],1.0),
                TrainingExample([-91.31429,-114.3718,-170.2887,-209.5479,51,24,32,10,-178.6222,-81.19243,-101.9763,-167.3187,-200.8598,47,23,31,10,-179.9829],0.0),
                TrainingExample([-42.08201,-57.71526,-56.0271,-73.74921,30,16,18,4,-144.6631,-41.72422,-53.60718,-74.16974,-80.0265,30,15,18,5,-141.0959],1.0),
                TrainingExample([-41.72422,-53.60718,-74.16974,-80.0265,30,15,18,5,-141.0959,-42.08201,-57.71526,-56.0271,-73.74921,30,16,18,4,-144.6631],0.0),
                TrainingExample([-27.97045,-42.6892,-62.42385,-72.11889,22,13,13,2,-138.0798,-37.23659,-52.0332,-47.41673,-59.97505,29,13,15,4,-147.1644],1.0),
                TrainingExample([-37.23659,-52.0332,-47.41673,-59.97505,29,13,15,4,-147.1644,-27.97045,-42.6892,-62.42385,-72.11889,22,13,13,2,-138.0798],0.0),
                TrainingExample([-42.23076,-57.38282,-80.06391,-91.61021,24,15,19,6,-146.6433,-45.00954,-46.97235,-108.6906,-114.4459,18,19,20,3,-133.4822],1.0),
                TrainingExample([-45.00954,-46.97235,-108.6906,-114.4459,18,19,20,3,-133.4822,-42.23076,-57.38282,-80.06391,-91.61021,24,15,19,6,-146.6433],0.0)]

    # Create a video and image writer
    fig, writer = generate_writer()


    print "Generating an image of the neural network loaded"
    network.load("dnn.10.__AllNodes__.txt")
    network.draw()
    take_still("neural_network_before.png")

    # Generate a video of the neural network learning
    print "Generating a video of the neural network learning."


    with writer.saving(fig, parameters.video_file_name, 100):
        for e, example in enumerate(examples):
            print e, example
            network.think(example.inputs)
            network.draw()
            annotate_frame(example)
            writer.grab_frame()
        print "Success! Open the file " + parameters.video_file_name + " to view the video."




