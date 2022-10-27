import numpy as np


class TransformationFactory:
    """
    Factory for creating transformations from 'transformation info'.
    """
    @staticmethod
    def get_transformation(transformation_info: dict, nb_allowed_values):
        """
        Method for creating transformation class instances.
        Transformation info is a dictionary with key 'name' that should be either 'negate2d' or 'polynomial'. Additionally,
        for polynomial transformation a 'degree' is needed, therefore this dictionary also has 'degree' key when using
        polynomial transformation.
        :param transformation_info: Dictionary that has information about the 'name' of the transformation and additional information if needed.
        :param nb_allowed_values: Number of allowed values. It is a number of how many possible values can there be for a
                specific values for later transformations. This is used to adjust possible random shifting values to better
                fit the values to be transformed.
        :return: Returns an instance of the transformation defined with 'transformation_info', or None if the arguments were not valid.
        """
        if transformation_info['name'] == "polynomial":
            return PolynomialTransformation(transformation_info['degree'], nb_allowed_values)
        if transformation_info['name'] == "negate2d":
            return Negate2dTransformation()
        return None


class PolynomialTransformation:
    """
    Polynomial transformation. Defined by the degree.
    """
    def __init__(self, degree, nb_allowed_values):
        """
        Initializes the polynomial transformation.
        Sets the coefficients to a random number from a given interval.
        :param degree: Degree of the polynomial.
        :param nb_allowed_values: Number of possible values for value to be transformed later.
        """
        self.degree = degree
        #self.coefficients = np.random.uniform(low=0.8, high=1.2, size=(degree + 1,))
        self.coefficients = np.zeros(degree+1)
        self.coefficients[degree] = np.random.uniform(-0.2 * nb_allowed_values, 0.2 * nb_allowed_values)
        self.coefficients[degree-1] = np.random.uniform(-2, 2)
        for i in range(degree-2, -1, -1):
            factor = (0.5**(degree-2-i))
            self.coefficients[i] = np.random.uniform(-factor, factor)

    def transform(self, x):
        return np.polyval(self.coefficients, x)


class Negate2dTransformation:
    def transform(self, x):
        return (-1) * x
