import os

##################################################################################################
# Script for running polynomial transformations with different coefficient intervals and degrees #
##################################################################################################

number = 'bigNb'
# number = 'smallerNb'
# number = 'smallNb'
global_trans = ['global', 'withoutglobal'] #
root = './folder/'

script_name = 'transform.py'
plot_script = 'plot_faults_PASS.py'

orig_json = 'random.json'

degrees = [1, 2, 3]
modes = ['clip', 'modulo', 'scale']
run = '1'
for gl in global_trans:
    folder = root + '-'.join([number, gl])
    print(folder)
    for degree in degrees:
        for mode in modes:
            cfg_name = '_'.join(['parameters', 'polynomial', str(degree), mode]) + '.cfg'
            print(cfg_name)
            os.system(' '.join([script_name, folder, orig_json, cfg_name, run]))
            file_name = '_'.join([orig_json, cfg_name, 'polynomial', str(degree), run]) + '.json'
            print(file_name)
            os.system(' '.join([plot_script, folder, file_name, cfg_name]))
