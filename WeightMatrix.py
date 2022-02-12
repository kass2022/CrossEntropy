from numpy import zeros, sqrt, sum
import copy

class data_weight:

    def __init__(self, data):

        self.data = data
        self.data_weight = zeros(self.data.shape)

        self.num = len(self.data)
        self.weight = zeros(self.num)
        
    def Zhdanov(self, sensitivity_matrix):
        for i in range(self.num):
            self.weight[i] = sqrt(sum(sensitivity_matrix[i, :] ** 2))

    def weighting(self):
        self.data_weight = self.data * self.weight


class model_weight:

    def __init__(self, solution):

        self.solution = solution
        self.solution_weight = zeros(self.solution.shape)

        self.num = len(self.solution)
        self.weight = zeros(self.num)

    def Zhdanov(self, sensitivity_matrix):
        for i in range(self.num):
            self.weight[i] = sqrt(sum(sensitivity_matrix[:, i] ** 2))

    def weighting(self):
        self.solution_weight = self.solution * self.weight

    def unweighting(self, solution_weight):
        self.solution = solution_weight / self.weight
   

class sensitivity_matrix_weight:
    
    def __init__(self, sensitivity_matrix, data_weight, solution_weight):

        self.sensitivity_matrix = sensitivity_matrix
        self.sensitivity_matrix_weight = zeros(self.sensitivity_matrix.shape)
        self.data_weight = data_weight
        self.solution_weight = solution_weight

        self.row = self.sensitivity_matrix.shape[0]
        self.column = self.sensitivity_matrix.shape[1]

    def weighting(self):

        for i in range(self.row):
            self.sensitivity_matrix_weight[i,:] = self.sensitivity_matrix[i,:] * self.data_weight[i]
        for i in range(self.column):
            self.sensitivity_matrix_weight[:,i] = self.sensitivity_matrix_weight[:,i] / self.solution_weight[i]


class weight:

    def __init__(self, solution, data, sensitivity_matrix):
        
        self.data_weighting = data_weight(data)
        self.solution_weighting = model_weight(solution)
        self.sensitivity_matrix = sensitivity_matrix

        self.dw = None
        self.mw = None
        self.Aw = None

    def weighting_data(self, Index):

        if Index == 'Zhdanov':
            self.data_weighting.Zhdanov(self.sensitivity_matrix)
            self.data_weighting.weighting()

    def weighting_solution(self, Index):

        if Index == 'Zhdanov':
            self.solution_weighting.Zhdanov(self.sensitivity_matrix)
            self.solution_weighting.weighting()

    def weighting_sensitivity_matrix(self):
        
        self.sensitivity_matrix_weighting = sensitivity_matrix_weight(self.sensitivity_matrix,
                                                                      self.data_weighting.weight,
                                                                      self.solution_weighting.weight)
        self.sensitivity_matrix_weighting.weighting()       

    def weighting(self, Index1, Index2):

        self.weighting_data(Index1)
        self.weighting_solution(Index2)
        self.weighting_sensitivity_matrix()
        
        self.dw = copy.copy(self.data_weighting.data_weight)
        self.mw = copy.copy(self.solution_weighting.solution_weight)
        self.Aw = copy.copy(self.sensitivity_matrix_weighting.sensitivity_matrix_weight)



