import pandas as pd
from io_functions import *
import sys

from plot_reversed_plots import read_reversed_json
from svcca import cca_core
from scipy import stats
import collections


def get_kbl(classes, tclasses):
    orig = collections.Counter(classes)
    trans = collections.Counter(tclasses)
    origs = [orig[key] for key in orig.keys()]
    trans = [trans[key] for key in orig.keys()]
    # If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=axis).
    # if not normalized, the function also does that
    return stats.entropy(origs, trans)


def get_kbl_values(values, tvalues, allowed_values):
    orig = collections.Counter(values)
    trans = collections.Counter(tvalues)
    origs = [orig.get(val, 1e-6) for val in allowed_values]
    transs = [trans.get(val, 1e-6) for val in allowed_values]
    # If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=axis).
    # if not normalized, the function also does that
    return stats.entropy(origs, transs)


def nparange_from_param_info(param_info, param_name):
    info = param_info[param_name]
    return np.arange(info['min'], info['max'] + info['step'], info['step'])


def main(x, y):
    """
    Script to compare the original data and transformed data.
    Call the script in command line: <script> <root_folder> <orig_faults_file> <transformed_faults_file> <config_file>.
    Script is the name of this script.
    Root folder is the folder where output files will be saved, and from where the following files are taken.
    For all the io functions, root + file name is used.
    Faults file is the file with fault injection parameters that needs to be transformed.
    Config file is the file with descriptions of the transformation to be used, handling of transformed values that 
    are out of scope, and parameters.
    For the formats of these files please look at the documentation of the io_functions.py script.
    
    Prints number of different fault classes and percentages.
    Prints out cca score.
    Prints out Kullback-Leibler divergence score.
    """
    if len(sys.argv) != 5:
        sys.exit(
            "Wrong number of command line arguments. Write <script> <root_folder> <orig_faults_file> "
            "<transformed_faults_file> <config_file>.")

    root = sys.argv[1]
    original = sys.argv[2]
    transformed = sys.argv[3]
    config_file = sys.argv[4]

    # metadata, header, results = read_faults_file(root, original)
    # tmetadata, theader, tresults = read_faults_file(root, transformed, groupby=False)
    header, results = read_reversed_json(root, original)
    theader, tresults = read_reversed_json(root, transformed)

    # assert ((header == theader))
    # assert (results.shape == tresults.shape)

    results_df = pd.DataFrame(results, columns=list(header)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))
    tresults_df = pd.DataFrame(tresults, columns=list(theader)).apply(
        lambda column: pd.to_numeric(column, errors='ignore'))

    _, parameters_info = read_config_file(root, config_file)
    inputs = list(set(header) & set(parameters_info.keys()))

    print('KBL without grouping double points:')
    # print(get_kbl(results_df['STATUS'], tresults_df['STATUS']))

    print('KBL with double points grouped:')
    grouped = results_df.groupby(inputs).agg({'STATUS': ','.join})
    tgrouped = tresults_df.groupby(inputs).agg({'STATUS': ','.join})
    classes = np.array(
        [fault_type.changing.name if len(set(c.split(','))) != 1 else c.split(',')[0] for c in grouped['STATUS']])
    tclasses = np.array(
        [fault_type.changing.name if len(set(c.split(','))) != 1 else c.split(',')[0] for c in tgrouped['STATUS']])
    print(get_kbl(classes, tclasses))

    # df = pd.DataFrame(
    #     results_df.groupby(inputs)['STATUS'].apply(lambda x: list(x)[0] if len(set(x)) == 1 else 'changing')).reset_index()
    experiments_index = np.where(results_df['STATUS'] == results_df['STATUS']) #     experiments_index = np.where(results_df['STATUS'] == results_df['STATUS']) # ##
    original_params = results_df[[x, y]].values[experiments_index].T
    experiments_index = np.where(tresults_df['STATUS'] == tresults_df['STATUS']) #!= 'Pass') #  #
    transformed_params = tresults_df[[x, y]].values[experiments_index].T
    # res = cca_core.get_cca_similarity(original_params[:, :(transformed_params.shape[1])], transformed_params, verbose=0)
    res = cca_core.get_cca_similarity(original_params, transformed_params, verbose=0)
    print("cca", res['cca_coef1'])
    print("cca", np.mean(res['cca_coef1']))
    # original_params = results_df[inputs].values.T
    # transformed_params = tresults_df[inputs].values.T
    # res = cca_core.get_cca_similarity(original_params, transformed_params, verbose=0)
    # print(inputs)
    # print("cca", res['cca_coef1'])
    # print("avg cca", np.mean(res['cca_coef1']))

    akld = get_kbl_values(results_df[x].values, tresults_df[x].values, nparange_from_param_info(parameters_info, x))
    bkld = get_kbl_values(results_df[y].values, tresults_df[y].values, nparange_from_param_info(parameters_info, y))
    print('kld for a and b', akld, bkld)
    avgkld = np.mean([akld, bkld])
    print('avg kld', avgkld)
    return get_kbl(classes, tclasses), avgkld # , np.mean(res['cca_coef1'][3:])


if __name__ == "__main__":
    main('x', 'y')
