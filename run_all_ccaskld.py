import os, sys
import compare
import numpy as np
import pandas as pd

########################################################
# Script for calculating KLD and CCA on different data #
########################################################

# number = 'bigNb'
# number = 'smallerNb'
# numbers = ['bigNb', 'smallerNb']
# global_trans = ['global', 'withoutglobal']
root = './pics_to_transform/'

script_name = 'compare.py'

orig_json = 'evolvingGA_picek_batina.png.json'
# orig_json = 'maldini.png.json'
# orig_json = 'plus_shape.json'


# degrees = [1, 2, 3]
# modes = ['clip', 'modulo', 'scale']

# all_columns = []
# for nb in numbers:
#     for gl in global_trans:
#         folder = root + '-'.join([nb, gl])
#         print(folder)
#         columnkld = []
#         columncca = []
#         for degree in degrees:
#             for mode in modes:
#                 cfg_name = '_'.join(['parameters', 'polynomial', str(degree), mode]) + '.cfg'
#                 # print(cfg_name)
#                 file_name = '_'.join([orig_json, cfg_name, 'polynomial', str(degree)]) + '.json'
#                 # print(file_name)
#                 sys.argv = [script_name, folder, orig_json, file_name, cfg_name]
#                 kld1, kld2, cca = compare.main()
#                 columnkld.append(np.round(kld2, 4))
#                 columncca.append(np.round(cca, 4))
#         all_columns.append(columnkld)
#         all_columns.append(columncca)
#         print(np.min(columncca))
#         print(np.max(columncca))
# transposed = np.array(all_columns).T
# # np.savetxt(root+"foo.csv", transposed, delimiter=";")
# pd.DataFrame(transposed).to_csv(root+"foo.csv")

# files = [#'maldini.png.json_x_0.28_0.99_y_0.01_0.99_angle_165.json',
#          # 'maldini.png.json_x_0.05_0.32_y_0.55_0.93_angle_310.json',
#          #  'maldini.png.json_x_0.63_0.93_y_0_1_angle_172.json',
#          #  'maldini.png.json_x_0.72_0.94_y_0.69_0.91_angle_313.json',
#          # 'maldini.png.json_polynomial_1_clip_w_glob.json',
#         # 'maldini.png.json_polynomial_1_clip_wo_glob.json',
#     # 'maldini.png.json_polynomial_1_modulo_w_glob.json',
#     # 'maldini.png.json_polynomial_1_modulo_wo_glob.json',
#     # 'maldini.png.json_polynomial_1_scale_w_glob.json',
#     # 'maldini.png.json_polynomial_1_modulo_w_glob.json'
#     'maldini.png.json_x_0.28_0.99_y_0.01_0.99_angle_165.json',
#     'maldini.png.json_x_0.05_0.32_y_0.55_0.93_angle_310.json'
# ]

#
# files = [ #'plus_shape.json_x_0_0.97_y_0.01_0.95_angle_239.json',
#          'plus_shape.json_x_0.14_0.56_y_0.08_0.28_angle_15.json',
#          'plus_shape.json_x_0.17_0.99_y_0.01_1_angle_134.json',
#          ]

files = [ #'evolvingGA_x_0.23_0.99_y_0_0.99_angle_103.json',
         # 'evolvingGA_x_0.59_0.81_y_0.24_0.87_angle_212.json',
         # 'evolvingGA_x_0.05_0.97_y_0.52_0.96_angle_28.json',
    'evolvingGA_angle_-53_x_38_38_y_27_33.json',
    'evolvingGA_angle_-42_x_33_69_y_33_20.json',
    'evolvingGA_angle_-63_x_50_21_y_32_36.json',
    'evolvingGA_angle_-6_x_20_75_y_31_47.json'
         ]



# cfg_name = 'maldini_modulo.cfg'
# cfg_name = 'modulo_F_plus.cfg'
cfg_name = 'modulo_T_PB.cfg'
print(cfg_name)

all_columns = []
columnkld = []
for file_name in files:

    columncca = []

    print(file_name)
    sys.argv = [script_name, root, orig_json, file_name, cfg_name]
    # kld1, kld2, cca = compare.main('x', 'y')
    # kld1, kld2 = compare.main('A', 'B')
    kld1, kld2 = compare.main('x', 'y')
    columnkld.append(np.round(kld2, 4))
    # columncca.append(np.round(cca, 4))
all_columns.append(columnkld)
# all_columns.append(columncca)
# print(np.min(columncca))
# print(np.max(columncca))
print(all_columns)
transposed = np.array(all_columns).T
# np.savetxt(root+"foo.csv", transposed, delimiter=";")
pd.DataFrame(transposed).to_csv(root+orig_json+'.csv')