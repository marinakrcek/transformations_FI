import sys
from parameters import *
from io_functions import *
import pandas as pd
import numpy as np
from plot_reversed_plots import read_reversed_json


class whichFaults(Enum):
    only_pass = 0
    all = 1
    only_interesting = 2


def faults_for_transformations(results, which_faults=whichFaults.only_interesting):
    """
    Predicate function that returns a list of bool values that match elements in the arguments 'results'
    that do not have 'fault class' output as 'PASS'. This helps getting fault injection parameter sets
    that need to be transformed.
    :param which_faults:
    :param results: dataframe with all fault injection parameters and output values.
    :return: bool array with True for parameters that need to be transformed
    """
    if which_faults == whichFaults.only_interesting:
        return results['STATUS'] != 'Pass'
    if which_faults == whichFaults.only_pass:
        return results['STATUS'] == 'Pass'
    return results['STATUS'] == results['STATUS']


def all_columns_same(df, row, columns):
    """
    Predicate function to check which rows of dataframe 'df' have the same values in the columns with column names
    given in 'columns' argument as 'row' series.
    :param df: dataframe in which we look for rows with same values as in series 'row'
    :param row: series that we look for in the dataframe 'df'
    :param columns: List of column names that have to be equal in rows of 'df' with the series 'row'.
    :return: List of bool values. Element of the list is True if that row in 'df' has equal values as 'row', and False otherwise.
    """
    res = True
    for c in columns:
        res = res & (df[c] == row[c])
    return res


def transform(metadata, header, results, transformation_info, parameters_info, which_faults, set_as_pass_interesting_points=False):
    """
    Function that transforms the parameters of the fault injection.
    For multiple same parameters injections, transformed values will also be the same.
    It is assumed that the input values are always numeric.
    :param metadata: Metadata holds information about device, but also fixed parameters for that specific fault injection analysis.
    :param header: Header holds information about the input parameter names and output names.
    :param results: 2D array with columns according to the header and rows as multiple fault injections.
    :param transformation_info: Dictionary with 'name' for transformation name and additional keys for specific values needed for
    specific transformations.
    :param parameters_info: Dictionary with parameter info, such as hardware limitations (intervals) for specific parameters.
    :return: Returns metadata, header and transformed values in shape of the given results.
    """
    # fixed parameters: scaling will not work for them, so we do modulo for them
    fixed_parameters = ParameterSet.get_parameterset_from_dict(metadata, parameters_info)
    if fixed_parameters.parameters:
        fixed_parameters.transform(transformation_info, fixed_params=True)
        fixed_parameters = ParameterSet.global_transform_sets(transformation_info, {0: fixed_parameters},
                                                              fixed_params=True)
        metadata.update(fixed_parameters[0].to_dict())

    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    inputs = list(set(header) & set(parameters_info.keys()))
    experiments_index = np.where(faults_for_transformations(results_df, which_faults))
    params_to_transform = results_df.iloc[experiments_index][inputs].drop_duplicates(subset=inputs, keep='first')

    param_sets = {}
    for index, row in params_to_transform.iterrows():
        param_set = ParameterSet.get_parameterset_from_dict(row.to_dict(), parameters_info)
        param_set.transform(transformation_info)
        param_sets[index] = param_set

    if ParameterSet.global_transf:
        param_sets = ParameterSet.global_transform_sets(transformation_info, param_sets)
    if ParameterSet.handle_out_of_scope == "scale":
        ParameterSet.scale(param_sets)
    transformed_param_sets = param_sets

    if set_as_pass_interesting_points:
        for i, tps in transformed_param_sets.items():
            data = tps.to_dict()
            data['STATUS'] = results_df.loc[i, 'STATUS']
            results_df = results_df.append(pd.Series(data), ignore_index=True)
        results_df.loc[experiments_index[0], 'STATUS'] = 'Pass'
        return metadata, header, results_df.values

    for index, row in results_df.iterrows():
        trsindex = index
        if index not in params_to_transform.index.values:
            trsindex = params_to_transform.index[all_columns_same(params_to_transform, row, inputs)]
            if trsindex.size == 0:
                continue
            trsindex = trsindex[0]
        results_df.loc[index, inputs] = transformed_param_sets[trsindex].to_series()

    return metadata, header, results_df.values


def transform_add_transformed(metadata, header, results, transformation_info, parameters_info, which_faults):
    # not replacing, just adding transformed values to the set
    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    inputs = list(set(header) & set(parameters_info.keys()))
    experiments_index = np.where(faults_for_transformations(results_df, which_faults))
    params_to_transform = results_df.iloc[experiments_index][inputs + ['STATUS']].drop_duplicates(subset=inputs,
                                                                                                  keep='first')

    param_sets = {}
    for index, row in params_to_transform.iterrows():
        param_set = ParameterSet.get_parameterset_from_dict(row.to_dict(), parameters_info)
        param_set.transform(transformation_info)
        param_sets[index] = param_set

    # if ParameterSet.global_transf:
    #     param_sets = ParameterSet.global_transform_sets(transformation_info, param_sets)
    if ParameterSet.handle_out_of_scope == "scale":
        ParameterSet.scale(param_sets)
    transformed_param_sets = param_sets

    for i, tps in transformed_param_sets.items():
        data = tps.to_dict()
        data['STATUS'] = results_df.loc[i, 'STATUS']
        results_df = results_df.append(pd.Series(data), ignore_index=True)
    results_df.loc[experiments_index[0], 'STATUS'] = 'Pass'

    return metadata, header, results_df.values


def transform_2d(metadata, header, results, parameters_info, set_as_pass_interesting_points, which_faults, border=None):
    ## main 2d transformations
    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    inputs = list(set(header) & set(parameters_info.keys()))
    experiments_index = np.where(faults_for_transformations(results_df, which_faults))
    params_to_transform = results_df.iloc[experiments_index][inputs].drop_duplicates(subset=inputs, keep='first')

    param_sets = {}
    intervals_for_scale = {}
    for index, row in params_to_transform.iterrows():
        param_set = ParameterSet.get_parameterset_from_dict(row.to_dict(), parameters_info)
        if not param_sets:
            for p in param_set.parameters:
                if not p.for_transform:
                    continue
                nb_values = len(p.allowed_values)
                int_size = np.random.choice(np.arange(int(0.2 * nb_values), nb_values + 1))
                lower_bound = np.random.choice(np.arange(0, nb_values - int_size + 1))
                intervals_for_scale[p.name] = (lower_bound, lower_bound + int_size)  # this works ok with np.arange
                print('scale+translation', p.name, lower_bound, nb_values, int_size, lower_bound / nb_values,
                           (lower_bound + int_size) / nb_values)
                print('scale+translation', p.name, list(intervals_for_scale.values()), np.array(list(intervals_for_scale.values()))/nb_values)
                #     continue
        param_sets[index] = param_set

    indexes = [[p.index for p in pset.parameters] for pset in param_sets.values()]
    mins = np.amin(indexes, axis=0)
    maxs = np.amax(indexes, axis=0)
    pset = list(param_sets.values())[0]
    for i in range(0, len(pset.parameters)):
        p = pset.parameters[i]
        if p.name == 'x':
            intervals_for_scale[p.name] = (mins[i], maxs[i]+1)
        elif p.name == 'y':
            intervals_for_scale[p.name] = (mins[i], maxs[i]+1)
    # angle = np.random.choice(np.arange(-80, 30))
    angle = np.random.choice(np.arange(0, 360))
    ParameterSet.rotate(param_sets, angle)
    print('rotate angle', angle)

    ParameterSet.scale_to(param_sets, intervals_for_scale)
    transformed_param_sets = param_sets

    if set_as_pass_interesting_points:
        for i, tps in transformed_param_sets.items():
            data = tps.to_dict()
            data['STATUS'] = results_df.loc[i, 'STATUS']
            results_df = results_df.append(pd.Series(data), ignore_index=True)
        results_df.loc[experiments_index[0], 'STATUS'] = 'Pass'
        return metadata, header, results_df.values, border

    for index, row in results_df.iterrows():
        trsindex = index
        if index not in params_to_transform.index.values:
            trsindex = params_to_transform.index[all_columns_same(params_to_transform, row, inputs)]
            if trsindex.size == 0:
                continue
            trsindex = trsindex[0]
        results_df.loc[index, inputs] = transformed_param_sets[trsindex].to_series()
    return metadata, header, results_df.values, border


def scale_border_wise(header, results, parameters_info):
    #for voltage glitching (assumptions)
    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    inputs = list(set(header) & set(parameters_info.keys()))
    experiments_index = np.where(faults_for_transformations(results_df, whichFaults.all))
    params_to_transform = results_df.iloc[experiments_index][inputs].drop_duplicates(subset=inputs, keep='first')

    ps = ParameterSet.get_parameterset_from_dict(params_to_transform.iloc[0].to_dict(), parameters_info)
    for p in ps.parameters:
        if p.name == 'x':
            x_lims = len(p.allowed_values)
        if p.name == 'y':
            y_lims = len(p.allowed_values)
    borderx = np.random.choice(np.arange(int(0.2*x_lims), int(0.8*x_lims)))
    bordery = np.random.choice(np.arange(int(0.2*y_lims), int(0.8*y_lims)))
    forscalex = np.random.choice(np.arange(int(0.2*x_lims), int(0.8*x_lims)))
    forscaley = np.random.choice(np.arange(int(0.2*y_lims), int(0.8*y_lims)))
    print(borderx, '->', forscalex, bordery, '->', forscaley)
    param_sets = {}
    for index, row in params_to_transform.iterrows():
        param_set = ParameterSet.get_parameterset_from_dict(row.to_dict(), parameters_info)
        for p in param_set.parameters:
            if p.name == 'x':
                if p.index < borderx:
                    p.scale_to(0, borderx, 0, forscalex)
                else:
                    p.scale_to(borderx, x_lims, forscalex, x_lims)
            elif p.name == 'y':
                if p.index < bordery:
                    p.scale_to(0, bordery, 0, forscaley)
                else:
                    p.scale_to(bordery, y_lims, forscaley, y_lims)
        param_sets[index] = param_set

    transformed_param_sets = param_sets
    for index, row in results_df.iterrows():
        trsindex = index
        if index not in params_to_transform.index.values:
            trsindex = params_to_transform.index[all_columns_same(params_to_transform, row, inputs)]
            if trsindex.size == 0:
                continue
            trsindex = trsindex[0]
        results_df.loc[index, inputs] = transformed_param_sets[trsindex].to_series()
    return header, results_df.values


def create_output_name(faults, config, transformation_info: dict, run):
    """
    Creates the output name from the given file names and transformation information so that from the name user is able to
    know which files where used for the transformation values in the file.
    :param faults: Faults file (original values)
    :param config: File with parameters information.
    :param transformation_info: Dictionary with information about transformation used.
    :return: String which will be the name of the output file.
    """
    return '_'.join(
        [get_filename(faults), get_filename(config), '_'.join(map(str, transformation_info.values())), run]) + '.json'


def randomize_int(header, results, parameters_info):
    ## 3d graph - intensity
    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    inputs = list(set(header) & set(parameters_info.keys()))
    experiments_index = np.where(faults_for_transformations(results_df, whichFaults.all))
    params_to_transform = results_df.iloc[experiments_index][inputs].drop_duplicates(subset=inputs, keep='first')

    ps = ParameterSet.get_parameterset_from_dict(params_to_transform.iloc[0].to_dict(), parameters_info)
    mapping = {}
    for p in ps.parameters:
        if p.name != 'intensity':
            continue
        indexes = np.arange(0, len(p.allowed_values))
        np.random.shuffle(indexes)
        for ii in np.arange(0, len(p.allowed_values)):
            # mapping[ii] = np.random.choice(np.arange(0, len(p.allowed_values)))
            mapping[ii] = indexes[ii]


    param_sets = {}
    for index, row in params_to_transform.iterrows():
        param_set = ParameterSet.get_parameterset_from_dict(row.to_dict(), parameters_info)
        for p in param_set.parameters:
            if p.name != 'intensity':
                continue
            p.change_index(mapping[p.index])
            # p.change_index(np.random.choice(np.arange(0, len(p.allowed_values))))
        param_sets[index] = param_set

    transformed_param_sets = param_sets
    for index, row in results_df.iterrows():
        trsindex = index
        if index not in params_to_transform.index.values:
            trsindex = params_to_transform.index[all_columns_same(params_to_transform, row, inputs)]
            if trsindex.size == 0:
                continue
            trsindex = trsindex[0]
        results_df.loc[index, inputs] = transformed_param_sets[trsindex].to_series()
    return header, results_df.values


if __name__ == "__main__":
    """
    Main script for transforming.
    Call the script using command line: <script> <root_folder> <faults_file> <config_file>
    Script is the name of this script. 
    Root folder is the folder where output files will be saved, and from where the following files are taken.
    For all the io functions, root + file name is used.
    Faults file is the file with fault injection parameters that needs to be transformed.
    Config file is the file with descriptions of the transformation to be used, handling of transformed values that 
    are out of scope, and parameters.
    For the formats of these files please look at the io_functions.py script.
    """
    import time

    if len(sys.argv) != 5:
        sys.exit("Wrong number of command line arguments. Write <script> <root_folder> <faults_file> <config_file>.")

    root = sys.argv[1]
    faults_file = sys.argv[2]
    config_file = sys.argv[3]
    run = sys.argv[4]
    replace_pass = True
    transformation_info, parameters_info = read_config_file(root, config_file)

    output_file = os.path.join(root, create_output_name(faults_file, config_file, transformation_info, run))
    # # simulated json file
    # metadata, header, results = read_faults_file(root, faults_file)
    # # reversed json file
    header, results = read_reversed_json(root, faults_file)
    metadata = dict()

    start = time.time()

    # tmetadata, theader, transformed = transform(metadata, header, results, transformation_info, parameters_info, whichFaults.only_interesting,replace_pass)
    # tmetadata, theader, transformed = transform_2d(tmetadata, theader, transformed, parameters_info, replace_pass, faults_to_transform2)
    tmetadata, theader, transformed, border = transform_2d(metadata, header, results, parameters_info,
                                                   replace_pass, whichFaults.only_interesting)
    # theader, transformed = scale_border_wise(theader, transformed, parameters_info)
    # tmetadata, theader, transformed = transform(tmetadata, theader, transformed, transformation_info, parameters_info,
    #                                             which_faults=whichFaults.only_pass)
    # tmetadata, theader, transformed = pass_around_interesting(tmetadata, theader, transformed, parameters_info)
    # tmetadata, theader, transformed, b, c = transform_2d(tmetadata, theader, transformed, parameters_info, replace_pass,
    #                                                whichFaults.only_pass, border, comb_for_transform)
    # tmetadata, theader, transformed = transform_add_transformed(metadata, header, results, transformation_info, parameters_info)
    # theader, transformed = randomize_int(header, results, parameters_info)
    end = time.time()
    # tmetadata = metadata
    # # simulated json file
    # save_transformed(output_file, tmetadata, theader, transformed)
    # # reversed json file
    save_reversed_transformed(output_file, tmetadata, theader, transformed)

    print(end - start)