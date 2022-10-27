import matplotlib.pyplot as plt
import pandas as pd
from io_functions import *
import sys


def plot_grid(grid, xlabel, ylabel, title, picture_name, x_min=0, x_max=1, y_min=0, y_max=1):
    """
    Function that plots the 2D grid.
    :param grid: Grid with Enum values for fault classes.
    :param xlabel: Name for x label
    :param ylabel: Name for y label
    :param title: Title of the plot
    :param picture_name: File name to save the plot to.
    :param x_min: Minimum value for x axis
    :param x_max: Maximum value for x axis
    :param y_min: minimum value for y axis
    :param y_max: maximum value for y axis
    """
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    normals = np.where(grid == fault_type.normal.value)
    resets = np.where(grid == fault_type.reset.value)
    changes = np.where(grid == fault_type.changing.value)
    success = np.where(grid == fault_type.success.value)

    plt.scatter(normals[0] / grid.shape[0], normals[1] / grid.shape[1], c="g", marker="x", label="NORMAL", s=5)
    plt.scatter(changes[0] / grid.shape[0], changes[1] / grid.shape[1], c="y", marker="D", label="CHANGING", s=5)
    plt.scatter(resets[0] / grid.shape[0], resets[1] / grid.shape[1], c="b", marker="o", label="RESET", s=5)
    plt.scatter(success[0] / grid.shape[0], success[1] / grid.shape[1], c="r", marker="s", label="SUCCESS", s=5)

    plt.legend(loc='lower left')
    plt.savefig(picture_name, format='png')
    plt.close()


def fault_class_to_value(fault_class: str):
    """
    Function that translates between string representation of the fault class to a
    Enum number for a fault class for easier plotting.
    :param fault_class: string 'normal', 'reset', 'changing' or 'success'
    :return: Enum value corresponding to the string value of the fault class.
    """
    if fault_class == fault_type.normal.name:
        return fault_type.normal.value
    if fault_class == fault_type.reset.name:
        return fault_type.reset.value
    if fault_class == fault_type.changing.name:
        return fault_type.changing.value
    if fault_class == fault_type.success.name:
        return fault_type.success.value
    raise ValueError("Unknown fault type: ", fault_class)


def fault_class(fault_value_prev, fault_class_new: str):
    """
    Returns a fault class based on previous fault class.
    This is needed for the fault points which were tested more than once.
    If previously we do not have information (-1), then the new fault class is returned.
    If the new fault class is the same as previous, then this fault class is returned.
    If the previous fault class and the new one do not match, then the fault class is 'changing'.
    :param fault_value_prev: Previous fault class
    :param fault_class_new: New fault class
    :return: Fault class based on information about the previous fault class and the new one.
    """
    if fault_value_prev == -1:
        return fault_class_to_value(fault_class_new)
    if fault_class_to_value(fault_class_new) == fault_value_prev:
        return fault_value_prev
    return fault_type.changing.value


def to_index(value, info):
    """
    Function that gives an index that corresponds to the given value.
    It calculates from possible values for the given parameter (from parameter information) at which index in the list of
    the possible values would the given value be.
    :param value: Values for which user requires the index in the list of possible values.
    :param info: Information about the specific parameter (min, max, step, ...)
    :return: index of the given value in the list of possible values for that parameter
    """
    if info.get('step') is None:
        return int(np.where(np.isclose(info.get('allowed_values'), value))[0])
    return int((value - info.get('min')) / info.get('step'))


def get_nb_axis_values(parameters_info, x_axis, y_axis):
    """
    Function to get number of possible values for the parameter on x axis and parameter on the y axis.
    :param parameters_info: Dictionary with parameter information.
    :param x_axis: Name of the parameter for x axis. String.
    :param y_axis: Name of the parameter for y axis. String.
    :return: Tuple with number of possible values for x, and y axis.
    """
    try:
        x_values = int((parameters_info[x_axis]['max'] - parameters_info[x_axis]['min']) / parameters_info[x_axis]['step'] + 1)
    except:
        x_values = len(parameters_info[x_axis]['allowed_values'])
    try:
        y_values = int((parameters_info[y_axis]['max'] - parameters_info[y_axis]['min']) / parameters_info[y_axis]['step'] + 1)
    except:
        y_values = len(parameters_info[y_axis]['allowed_values'])
    return x_values, y_values


def plot_faults(data, figure_name, x_axis, y_axis, parameters_info):
    """
    Function that plots the data to a figure_name using x_axis and y_axis for x and y axis in the plot.
    :param data: Dataframe from the faults file with all input and output values.
    :param figure_name: Name of the file to save the plot to.
    :param x_axis: Parameter name for x axis.
    :param y_axis: Parameter name for y axis.
    :param parameters_info: Information about parameters (min, max values...)
    """
    if x_axis not in parameters_info.keys() or y_axis not in parameters_info.keys():
        raise ValueError("There are no parameters with provided names for x and y axis: ", x_axis, y_axis)

    data[x_axis] = data[x_axis].astype(float, copy=False, errors='ignore')
    data[y_axis] = data[y_axis].astype(float, copy=False, errors='ignore')

    x_values, y_values = get_nb_axis_values(parameters_info, x_axis, y_axis)
    grid = np.full((x_values, y_values), -1)

    for index, row in data.iterrows():
        i, j = to_index(row[x_axis], parameters_info[x_axis]), to_index(row[y_axis], parameters_info[y_axis])
        grid[i][j] = fault_class(grid[i][j], row['fault class'])

    plot_grid(grid, xlabel='x', ylabel='y', title='faults', picture_name=figure_name)


def create_output_name(faults, config):
    """
    Creates the output name from the given file names. The file has '.png' extention.

    Feel free to adjust this according to your preferences in file naming, or how you see appropriate.
    :param faults: faults file (original or transformed)
    :param config: parameters file with information about parameter values
    :return: string representing a name for the plot png file
    """
    return 'plot_from_' + get_filename(faults) + '_' + get_filename(config) + '.png'


if __name__ == "__main__":
    """
    Script for plotting from the faults type of files.
    Call the script using command line: <script> <root_folder> <faults_file> <config_file>
    Script is the name of this script. 
    Root folder is the folder where output files will be saved, and from where the following files are taken.
    For all the io functions, root + file name is used.
    Faults file is the file with fault injection parameters that needs to be transformed.
    Config file is the file with descriptions of the transformation to be used, handling of transformed values that 
    are out of scope, and parameters.
    For the formats of these files please look at the io_functions.py script.
    """
    if len(sys.argv) != 4:
        sys.exit("Wrong number of command line arguments. Write <script> <root_folder> <faults_file> <config_file>.")

    root = sys.argv[1]
    faults_file = sys.argv[2]  # "faults.out"
    config_file = sys.argv[3]  # "parameters.cfg"

    figure_name = os.path.join(root, create_output_name(faults_file, config_file))
    metadata, header, results = read_faults_file(root, faults_file)
    results_df = pd.DataFrame(results, columns=list(header))
    # tuple here is transformation information (name and handling)
    _, parameters_info = read_config_file(root, config_file)
    # x and y are the parameters that go to x and y axis, respectively
    plot_faults(results_df, figure_name, 'x', 'y', parameters_info)
