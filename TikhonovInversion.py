import numpy as np
from Weight import VariablesWeight
from ConjugateGradient import CGSolver
import copy

class L2:
    def __init__(self):
        self.gradient = 0
        self.step = 0
    def update(self, alpha, m, Wm, direction, e):
        self.gradient = alpha * m * Wm
        # self.step = alpha * np.sum(direction**2)/Wm**2
        self.step = self.gradient / m * np.sum(direction**2)

class L0:
    def __init__(self):
        self.gradient = 0
        self.step = 0
    def update(self, alpha, m, Wm, direction, e):
        self.gradient = alpha * m * Wm**2 \
            * (m**2 + e**2 )**(-2)
        # self.step = alpha * np.sum(direction**2) \
        #     * e**2 * Wm**(-2) * ( (m / Wm)**2 + e**2 )**(-2)
        self.step = self.gradient / m * np.sum(direction**2)

class CrossEntropy:
    def __init__(self):
        self.gradient = 0
        self.step = 0
    def update(self, m, gradient, alpha):
        self.gradient = alpha * gradient
        self.step = alpha * gradient / m

class RRCGMethod:

    def __init__(self, sensitivity, data, model_initial):

        self.Variables = VariablesWeight(data, model_initial, sensitivity)   
        self.Variables.weight()  
        self.solver = CGSolver(self.Variables.S, self.Variables.d, self.Variables.m)
 
    def solve(self, epochs, error, alpha, Index = "L2", e = 0.1):

        self.alpha = alpha
        self.Index = Index
        self.e = e
        
        if Index == "L2":
            constrains = L2()
        if Index == "L0":
            constrains = L0()

        self.solver.epochs = epochs
        self.solver.error = error
        self.solver.count = 0

        self.solver.loss = np.zeros(self.solver.epochs)

        self.solver.computeGradient()
        self.solver.direction = copy.copy(self.solver.gradient)

        self.solver.computeBetaup()
        self.solver.computeStep()

        while True:

            self.solver.updateSoluation()
            self.solver.computeLoss()
            self.solver.count = self.solver.count + 1

            print("count = ", self.solver.count, ", loss = ", self.solver.loss[self.solver.count - 1])

            if self.solver.count == self.solver.epochs:
                break
            if self.solver.loss[self.solver.count] < self.solver.error:
                break

            constrains.update(self.alpha, self.solver.m, self.Variables.Wm, self.solver.direction, self.e)

            self.solver.computeBetadown()
            self.solver.computeGradient(constrains=constrains.gradient)

            self.solver.computeBetaup()

            self.solver.computeDirection()
            self.solver.computeStep(constrains=constrains.step)



def RRCGInv(sensitivity, data, model_initial, epochs, error, alpha, Index, e):
    solver = RRCGMethod(sensitivity, data, model_initial)
    solver.solve(epochs, error, alpha, Index, e)
    result = np.reshape(solver.solver.m,(len(model.x_model) - 1, len(model.y_model) - 1, len(model.z_model) - 1), order = 'F')
    return result 

if __name__ == "__main__":

    from Sensitivity import GraModel
    import matplotlib.pyplot as plt
    from ToolsFunction import Imshow

    ## forward anomaly
    model = GraModel(0, 500, 35*500, 
                    0, 500, 19*500, 
                    0, 500, 35*500,
                    0, 500, 19*500,
                    0, 500, 15*500)
    # model.property[12:15, 9:11, 3:6] = 1
    # model.property[17:20, 9:11, 3:6] = -1
    # model.property[22:25, 9:11, 4:7] = -1
    # model.property[27:30, 9:11, 3:6] = 1

    model.property[8:13, 9:11, 3:8] = 1
    model.property[22:27, 9:11, 3:10] = -1

    model.forward()

    S = model.sensitivity
    d = model.anomaly_vector
    m0 = np.zeros(model.property_vector.shape)

    # set inversion parameters
    epochs = 1000
    error = 0
    alpha = 1e-1
    Index = "L2"
    e = 0.1

    def exampleSolver():

        solver = RRCGMethod(S, d, m0)
        solver.solve(epochs, error, alpha, Index = Index, e = e)

        result = np.reshape(solver.solver.m,(len(model.x_model) - 1, len(model.y_model) - 1, len(model.z_model) - 1), order = 'F')
        
        ## show the results 
        plt.figure(figsize=(10, 8))

        figsize1 = 2
        figsize2 = 2
        slice_y = 10
        
        i = 1
        plt.subplot(figsize1,figsize2,i)
        Imshow(model.property[:,slice_y,:],"original model",inverse=False)
        i += 1
        plt.subplot(figsize1,figsize2,i)
        Imshow(result[:,slice_y,:],"result model",inverse=False)
        i += 1
        plt.subplot(figsize1,figsize2,i)
        Imshow(model.anomaly, "original anomaly",inverse=False)

        model.property = solver.solver.m
        model.forward()        
        
        i += 1
        plt.subplot(figsize1,figsize2,i)
        Imshow(model.anomaly, "result anomaly",inverse=False)

    def exampleSolveDirectly():
        result = RRCGInv(S, d, m0, epochs, error, alpha, Index, e)

        ## show the results 
        plt.figure(figsize=(10, 8))
        slice_y = 10
        Imshow(result[:,slice_y,:],"result model",inverse=False)



    exampleSolver()
    # exampleSolveDirectly()
    plt.show()