# What Do You See? Transforming Fault Injection Target Characterizations

Authors: Marina Krček

Source code for publication **What Do You See? Transforming Fault Injection Target Characterizations**

___________________________________________________________

### About the repository and code

Directory *svcca* is a copy of a repository from [https://github.com/google/svcca](https://github.com/google/svcca)
- used for CCA calculations

Script *compare.py*
- used to compare KLD and CCA for original and transformed data
- used by script *run_all_ccaskld.py* to automate it

Script *extract_points_from_plot.py*
- used to extract the information from target cartography figures
- for each picture, one needs to with a color picker get the RGB codes for the colors, that way we distinguish which pixel belongs to which class
- parameter intervals if known, we can scale to those intervals to get exact parameter values

Script *io_functions.py*
- script with function to read data from different format files
- mostly it is json, but reversed files have less information than a simulated case

Script *parameters.py*
- has a Parameter class which hold the name of the parameter, its allowed values, keeps the index, can connect index with a value, etc.
- has a ParameterSet class which is one parameter combination for the fault injection
- the polynomial transformations are called here per parameter and parametersets as they know the values and can do out-of-bounds methods, and keep the index and value inline

Script *plot_faults.py* \
Script *plot_faults_PASS.py*
- used for plotting
- has the 3d plot function
- same as *plot_faults.py*, but use different fault class naming

Script *plot_reversed_plots.py*
- used to read json file of reversed plots, and plot those
- for the case of EMFI (maldini) and voltage glitching (evolvingGA) we have different x and y-axis values, which then need to be adjusted in the source code

Script *run_all_ccaskld.py*
- used to get KLD and CCA for different transformations and save to a csv file

Script *run_diff_transformations.py*
- was used to run polynomial transformations with different degrees and out-of-bounds methods
- it is used to automate part of the job
- with this we also tested different coefficient values/intervals/expressions (smaller and larger values)

Script *transform.py*
- has a main function where we load the data and then call function to transform them and save in a json file
- there are multiple functions with different transformations we tested (2D transformations, transformations for voltage glitching where we have to keep the assumptions, etc.)
- in main we can choose which transformations we will use (can be combined)
- polynomial transformations however rely on parameter classes
-

Script *transformations.py*
- has classes for polynomial transformations
- here we adjust the coefficients' values/ranges/expressions


___________________________________________________________

If you use this source code, please consider citing:
```

```
