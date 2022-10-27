import os, json
import numpy as np
import pandas as pd
from io_functions import fault_type
import matplotlib.pyplot as plt


def read_reversed_json(root, filename):
    with open(os.path.join(root, filename), 'r') as fp:
        data = json.load(fp)
    return data['columns'], np.array(data['data'])


def plot_data(data, figure_name, x_axis, y_axis, parameters_info):

    x_data = np.array(data[x_axis].astype(float, copy=False, errors='ignore'))
    y_data = np.array(data[y_axis].astype(float, copy=False, errors='ignore'))
    Y = np.array(data['STATUS'])

    # plt.rcParams["figure.figsize"] = (10, 6)

    marker_size = 2
    classes = [fault_type.Pass.name, fault_type.Mute.name, fault_type.changing.name, fault_type.Fail.name]
    colors = ['g', 'b', 'orange', 'r']
    for fault_name, color in zip(classes, colors):
        indexes = np.where(Y == fault_name)
        plt.scatter(x_data[indexes], y_data[indexes], label=fault_name.lower(), c=color, s=marker_size)

    plt.legend(fontsize=15, markerscale=3.)
    # plt.xlabel('Glitch voltage (V)', fontsize=15)
    # plt.ylabel('Glitch length (ns)', fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)

    number_of_ticks = 5
    plt.xticks(ticks=np.linspace(parameters_info[x_axis]['min'], parameters_info[x_axis]['max'], number_of_ticks),
               labels=np.linspace(0, 1, number_of_ticks),
               fontsize=15)
    plt.yticks(ticks=np.linspace(parameters_info[y_axis]['min'], parameters_info[y_axis]['max'], number_of_ticks),
               labels=np.linspace(0, 1, number_of_ticks),
               fontsize=15)
    # plt.rcParams.update({'legend': 15})
    plt.gca().invert_yaxis() # for maldini
    plt.savefig(figure_name, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    root = './pics_to_transform/'
    # jsonfile = 'evolvingGA_picek_batina.png.json'
    jsonfile = 'maldini.png.json'
    columnnames, data = read_reversed_json(root, jsonfile)
    results_df = pd.DataFrame(data, columns=list(columnnames))

    x_axis = 'x'
    y_axis = 'y'
    parameters_info = dict()
    parameters_info['x'] = {'min': 0, 'max': 24, 'step': 0.05}
    parameters_info['y'] = {'min': 0, 'max': 24, 'step': 0.05}
    # parameters_info['x'] = {'min': -5, 'max': -0.05, 'step': 0.05}
    # parameters_info['y'] = {'min': 2, 'max': 150, 'step': 2}

    plot_data(results_df, root+jsonfile+'.png', x_axis, y_axis, parameters_info)

