import parameters
from matplotlib import pyplot, animation, rcParams


def generate_writer():
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=parameters.frames_per_second, metadata=parameters.metadata)
    fig = pyplot.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    pyplot.xlim(0, parameters.width)
    pyplot.ylim(0, parameters.height)
    axis = pyplot.gca()
    axis.set_axis_bgcolor('black')
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    rcParams['font.size'] = 12
    rcParams['text.color'] = 'white'
    return fig, writer

# def annotate_frame_simple():
#     pyplot.text(1, parameters.height - 1, "TODO")
# def annotate_frame():
#     pyplot.text(1, parameters.height - 1, "TODO" )
#


def annotate_frame(example):
    pyplot.text(1, parameters.output_y_position + 5, "Desired output:")
    pyplot.text(1, parameters.output_y_position + 4, str(example.output), fontsize=10)
    pyplot.text(1, parameters.output_y_position + 9, "Inputs:")
    pyplot.text(1, parameters.output_y_position + 8, str(example.inputs[0:9]), fontsize=10)
    pyplot.text(1, parameters.output_y_position + 7, str(example.inputs[9:18]), fontsize=10)



def error_bar(average_error):
    pyplot.text(parameters.error_bar_x_position, parameters.height - 1, "Average Error " + str(average_error) + "%")
    border = pyplot.Rectangle((parameters.error_bar_x_position, parameters.height - 3), 10, 1, color='white', fill=False)
    pyplot.gca().add_patch(border)
    rectangle = pyplot.Rectangle((parameters.error_bar_x_position, parameters.height - 3), 10 * average_error / 100, 1, color='red')
    pyplot.gca().add_patch(rectangle)


def take_still(image_file_name):
    pyplot.savefig(image_file_name)
