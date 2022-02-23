from Sensitivity import MagModel
import matplotlib.pyplot as plt
from ToolsFunction import Imshow
import numpy as np
from Regularization import VariablesWeight
from ConjugateGradient import CGSolver
import copy

class RRCGInv:

    def __init__(self, sensitivity, data, model_initial):

        self.Variables = VariablesWeight(data, model_initial, sensitivity)   
        self.Variables.weight()  
        self.solver = CGSolver(self.Variables.Sw, self.Variables.dw, self.Variables.mw)
 
    def computeConstraintGradient(self, Index, alpha):
        if Index == "L2":
            self.solver.gradient += alpha * self.solver.m / self.Variables.Wm
        # if Index == "L0":


    def computeConstraintStep(self, Index, alpha):
        if Index == "L2":
            self.solver.step_down += np.sum(alpha * self.solver.direction**2/self.Variables.Wm**2)
        self.solver.step = self.solver.beta_up / self.solver.step_down

    def solve(self, epochs, error, alpha, Index = "L2"):

        self.alpha = alpha
        self.Index = Index

        self.solver.epochs = epochs
        self.solver.error = error
        self.solver.count = 0

        self.solver.loss = np.zeros(self.solver.epochs)

        self.solver.computeGradient()
        self.computeConstraintGradient(self.Index, self.alpha)
        self.solver.direction = copy.copy(self.solver.gradient)

        self.solver.computeBetaup()
        self.solver.computeStep()
        self.computeConstraintStep(self.Index, self.alpha)

        while True:

            self.solver.updateSoluation()
            self.solver.computeLoss()

            self.solver.count = self.solver.count + 1

            print("count = ", self.solver.count, ", loss = ", self.solver.loss[self.solver.count - 1])

            if self.solver.count == self.solver.epochs:
                break

            if self.solver.loss[self.solver.count] < self.solver.error:
                break

            self.solver.computeBetadown()
            self.solver.computeGradient()

            self.computeConstraintGradient(self.Index, self.alpha)
            self.solver.computeBetaup()

            self.solver.computeDirection()
            self.solver.computeStep()

            self.computeConstraintStep(self.Index, self.alpha)

        self.Variables.mw = self.solver.m
        self.Variables.unweight()


if __name__ == "__main__":
    from Sensitivity import GraModel
    import matplotlib.pyplot as plt
    from ToolsFunction import Imshow
    import numpy as np
    from Regularization import VariablesWeight

    ## forward anomaly
    model = GraModel(0, 500, 35*500, 
                    0, 500, 19*500, 
                    0, 500, 35*500,
                    0, 500, 19*500,
                    0, 500, 15*500)
    model.property[7:9, 9:11, 3:6] = 1
    model.property[12:14, 9:11, 3:6] = 1
    model.property[17:19, 9:11, 3:6] = -1
    model.property[22:24, 9:11, 4:7] = -1
    model.property[27:29, 9:11, 3:6] = 1

    model.forward()

    S = model.sensitivity
    d = model.anomaly_vector
    m0 = np.zeros(model.property_vector.shape)

    ## Inverse the model 
    epochs = 100
    error = 0
    alpha = 1000
    solver = RRCGInv(S, d, m0)
    solver.solve(epochs, error, alpha, Index = "L2")

    result = np.reshape(solver.Variables.m,(len(model.x_model) - 1, len(model.y_model) - 1, len(model.z_model) - 1), order = 'F')
    
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
    i += 1
    # model.property = solver.Variables.m
    model.forward()
    plt.subplot(figsize1,figsize2,i)
    Imshow(model.anomaly, "result anomaly",inverse=False)
    plt.show()




