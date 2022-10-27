import pandas as pd
from transformations import *


class Parameter:
    """
    Class Parameter holds all the details about a specific parameter.
    Holds the name of the parameter,
    value of the parameter,
    it can hold allowed value (when step there), or
    it can have min and max values for the parameter with a step,
    from which in the initialization a list with allowed values is created.
    Also hold bool value of whether the parameter should be transformed or not.
    And it holds the index of the parameter value. Index is index in the allowed values list,
    which corresponds to the value of the parameter.
    """
    def __init__(self, name, value, info):
        """
        Initializes the Parameter class instance.
        :param name: Name of the parameter
        :param value: Value of the parameter
        :param info: dictionary with information about the given parameter.
        """
        self.name = name
        self.value = value
        if "allowed_values" in info.keys():
            self.allowed_values = info["allowed_values"]
            self.step = 0
        else:
            self.max = info["max"]
            self.min = info["min"]
            self.step = info["step"]
            self.allowed_values = list(np.arange(self.min, self.max + self.step, self.step).tolist())
        self.for_transform = bool(info["transform"])
        self.index = self.get_index(self.value)

    def get_index(self, value):
        """
        Function that gives an index that corresponds to the given value.
        It calculates from possible values for the given parameter (from parameter information) at which index in the list of
        the possible values would the given value be.
        :param value: Values for which user requires the index in the list of possible values.\
        :return: index of the given value in the list of possible values for that parameter
        """
        if self.step == 0:
            return int(np.where(np.isclose(self.allowed_values, value))[0])
        return int((value - self.min) / self.step)

    def get_value(self, index):
        """
        Function that returns a value from a given index.
        :param index: Index in the allowed values list, which corresponds to the value of the parameter.
        :return: Value of the parameter at the index of the allowed values for the parameter
        """
        if 0 <= index < len(self.allowed_values):
            return self.allowed_values[index]
        return index * self.step + self.min

    def transform(self, transformation, handle_out_of_scope):
        """
        Transforms the parameter according to the given instance of the transformation and
        handles out of scope values in a given way.
        Value and index of the parameter are changed.
        If using clip or modulo, the values are immediately valid.
        If using scaling, the values are invalid until the global transformation is done to
        do the scaling.
        :param transformation: Instance of a transformation class.
        :param handle_out_of_scope: String. It can be 'clip', 'modulo', or 'scale'.
        """
        self.index = int(np.round(transformation.transform(self.index)))
        if handle_out_of_scope == "scale":
            return  # do not change the index, nor the value; will done with scaling at the end
        if handle_out_of_scope == "clip":
            self.index = np.clip(self.index, 0, len(self.allowed_values) - 1)
        elif handle_out_of_scope == "modulo":
            self.index = self.index % len(self.allowed_values)
        self.value = self.allowed_values[self.index]

    def scale(self, min, max):
        """
        Function scales and updates, the value and index of the parameter, according to the given
        min and max values. It uses the initial allowed intervals stored in the instance of the class.
        No return value, parameter variables are updated.
        :param min: Min value for this parameter after transformations.
        :param max: Max value for this parameter after transformations.
        """
        # new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        # new_max is the last index of the allowed values, new_min is 0
        self.index = int(round(((self.index - min) / (max - min)) * (len(self.allowed_values) - 1)))
        self.value = self.allowed_values[int(self.index)]

    def scale_to(self, min, max, newmin, newmax):
        """
        Scales to provided new interval
        :param min: Min value for this parameter after transformations.
        :param max: Max value for this parameter after transformations.
        :param newmin:
        :param newmax:
        """
        # new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        self.index = int(round(((self.index - min) / (max - min)) * (newmax - 1 - newmin) + newmin))
        self.value = self.allowed_values[int(self.index)]

    def change_index(self, new_index):
        self.value = self.allowed_values[int(new_index)]
        self.index = new_index


class ParameterSet:
    """
    Class that holds multiple different parameters in a list and operates on the sets.
    """
    handle_out_of_scope = None
    global_transf = True

    # holds a list of parameters
    def __init__(self, parameters: list):
        """
        Takes a list of parameters and saves it.
        :param parameters: list of parameteres
        """
        self.parameters = parameters

    def transform(self, transformation_info: dict, fixed_params=False):
        """
        Performs local transformation on parameters of the set. Parameters are changed, no return values.
        For transformations on fixed parameters, set the bool value to True.
        For fixed parameters, scaling cannot be used because only one value is changed, there are no intervals (min and max).
        For fixed parameters, 'modulo' will be used.
        If scaling is used for handling out of scope, value and index values of parameters will be unusable until the global
        transformation is also executed. Scaling needs information about the minimum and maximum values after transfomations to
        be able to scale.
        :param transformation_info: dictionary with transformation for the transformation factory.
        :param fixed_params: bool value, has to be True when transformation is done for fixed parameters, False otherwise. Default is False.
        """
        handle_out_of_scope = 'modulo' if fixed_params else ParameterSet.handle_out_of_scope
        for p in self.parameters:
            if not p.for_transform:
                continue
            p.transform(TransformationFactory.get_transformation(transformation_info, len(p.allowed_values)), handle_out_of_scope)

    def transform2d(self, transformation_info: dict, index):
        to_negate = [p for p in self.parameters if p.for_transform]
        if len(to_negate) > 2:
            raise ValueError("For negate2d transformation, only two parameters should be used, not ", len(to_negate))
        transformation = TransformationFactory.get_transformation(transformation_info, -1)
        if ParameterSet.handle_out_of_scope != 'modulo':
            raise ValueError('With negate2d transformation the handling of negative values has to be modulo and not ', ParameterSet.handle_out_of_scope)
        if index == 2:
            for p in self.parameters:
                if not p.for_transform: continue
                p.transform(transformation, self.handle_out_of_scope)
        else:
            i = 0
            for p in self.parameters:
                if not p.for_transform: continue
                if i == index:
                    p.transform(transformation, self.handle_out_of_scope)
                i = i + 1

    def global_transform(self, transformation, fixed_params=False):
        """
        Performs global transformation on parameters of the set. Parameters are changed, no return values.
        For transformations on fixed parameters, set the bool value to True.
        For fixed parameters, scaling cannot be used because only one value is changed, there are no intervals (min and max).
        For fixed parameters, 'modulo' will be used.
        :param transformation: Already created transformation that will be used for all the parameters.
        :param fixed_params: bool value, has to be True when transformation is done for fixed parameters, False otherwise. Default is False.
        """
        handle_out_of_scope = 'modulo' if fixed_params else ParameterSet.handle_out_of_scope
        for p in self.parameters:
            if not p.for_transform:
                continue
            p.transform(transformation, handle_out_of_scope)

    def to_dict(self):
        """
        Converts parameter set instance to a dictionary.
        :return: A dictionary from the parameter set. With parameter name as key, and parameter value as value.
        """
        return {p.name: p.value for p in self.parameters}

    def to_series(self):
        """
        Converts parameter set instance to a pandas.Series.
        :return: A pandas.Series from the parameter set, holding information about parameter name and value.
        """
        return pd.Series(self.to_dict())

    @staticmethod
    def scale(parameter_sets: dict):
        """
        Scaling function. Takes a dictionary of parameter sets. Keys are indexes in the original data, and values are the
        parameter sets. No return values, scaling is done in place.
        :param parameter_sets: Dictionary with parameter sets.
        """
        indexes = [[p.index for p in pset.parameters] for pset in parameter_sets.values()]
        mins = np.amin(indexes, axis=0)
        maxs = np.amax(indexes, axis=0)
        assert (len(mins) == len(maxs) == len(indexes[0]))
        for pset in parameter_sets.values():
            for i in range(0, len(pset.parameters)):
                pset.parameters[i].scale(mins[i], maxs[i])

    @staticmethod
    def scale_to(parameter_sets: dict, intervals_for_scale):
        """
        Scaling function. Takes a dictionary of parameter sets. Keys are indexes in the original data, and values are the
        parameter sets. No return values, scaling is done in place.
        :param parameter_sets: Dictionary with parameter sets.
        """
        indexes = [[p.index for p in pset.parameters] for pset in parameter_sets.values()]
        mins = np.amin(indexes, axis=0)
        maxs = np.amax(indexes, axis=0)
        assert (len(mins) == len(maxs) == len(indexes[0]))
        for pset in parameter_sets.values():
            for i in range(0, len(pset.parameters)):
                addition = -mins[i] if mins[i] < 0 else 0
                newinterval = intervals_for_scale.get(pset.parameters[i].name, (mins[i]+addition, maxs[i]+addition))
                pset.parameters[i].scale_to(mins[i], maxs[i], newinterval[0], newinterval[1])

    @staticmethod
    def rotate(parameter_sets: dict, angle):
        """
        Function to rotate the whole shape.
        :param parameter_sets:
        :param angle:
        :return:
        """
        parameter_list = list(parameter_sets.values())[0].parameters
        index_for_transform = list(dict(sorted({parameter_list[i].name: i for i in range(0, len(parameter_list)) if parameter_list[i].for_transform}.items())).values())
        if len(index_for_transform) > 2:
            raise ValueError('Rotating can be done only when two parameters are for transformations.')
        angle = np.radians(angle)  # np.cos and np.sin work with radians only
        cosa = np.cos(angle)
        sina = np.sin(angle)
        xi = index_for_transform[0]
        yi = index_for_transform[1]
        xc = 0  # int(np.round(len(parameter_list[xi].allowed_values) / 2.0))
        yc = 0  # int(np.round(len(parameter_list[yi].allowed_values) / 2.0))
        for pset in parameter_sets.values():
            x_index = pset.parameters[xi].index
            y_index = pset.parameters[yi].index
            # x2 = cosβx1−sinβy1
            pset.parameters[xi].index = int(np.round((x_index - xc) * cosa - (y_index - yc) * sina + xc))
            # y2 = sinβx1 + cosβy1
            pset.parameters[yi].index = int(np.round((x_index - xc) * sina + (y_index - yc) * cosa + yc))

    @staticmethod
    def global_transform_sets(transformation_info: dict, parameter_sets: dict, fixed_params=False):
        """
        Function that transforms given parameter sets.
        For transformations on fixed parameters, set the bool value to True.
        For fixed parameters, scaling cannot be used because only one value is changed, there are no intervals (min and max).
        For fixed parameters, 'modulo' will be used.
        :param transformation_info: Transformation info in a dictionary.
        :param parameter_sets: Parameter sets as a dictionary. Indexes from original data are keys. Values are parameter sets.
        :param fixed_params: bool value. Should be set to True when transforming fixed parameters, False otherwise (which is the default value).
        :return: A new dictionary with transformed values.
        """
        nb_allowed_values = np.amin([len(p.allowed_values) for p in list(parameter_sets.values())[0].parameters if p.for_transform])
        global_transformation = TransformationFactory.get_transformation(transformation_info, nb_allowed_values)
        for index, pset in parameter_sets.items():
            pset.global_transform(global_transformation, fixed_params)
        if ParameterSet.handle_out_of_scope == "scale" and not fixed_params:
            ParameterSet.scale(parameter_sets)
        return parameter_sets

    @staticmethod
    def get_parameterset_from_dict(data: dict, parameters_info: dict):
        """
        Creates parameter set from a given dictionary 'data' and parameter information.
        :param data: Dictionary with parameter names as keys, and their values as values of dictionary.
        :param parameters_info: Dictionary with parameter information (min, max, step...)
        :return: Instance of a parameter set class from a given dictionary 'data'.
        """
        ParameterSet.handle_out_of_scope = parameters_info.get('out_of_scope', 'clip')
        ParameterSet.global_transf = parameters_info.get('global_transf', False)
        parameters = []
        for (name, value) in data.items():
            if name not in parameters_info.keys():
                continue
            parameters.append(Parameter(name, value, parameters_info[name]))
        return ParameterSet(parameters)
