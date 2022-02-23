import numpy as np
import copy

class CGSolver:

    def __init__(self, sensitivity, data, model_initial):

        self.S = sensitivity
        self.d = data
        self.m = model_initial

        self.gradient = None
        self.direction = None

        self.step = 0
        self.step_down = None
        self.beta_down = 0
        self.beta_up = 0

    def computeGradient(self):
        self.gradient = self.S.T.dot(self.S.dot(self.m)) - self.S.T.dot(self.d)

    def computeStep(self):
        self.step_down = np.sum((self.S.dot(self.direction)) ** 2)
        self.step = self.beta_up / self.step_down

    def computeDirection(self):
        self.direction = self.gradient + self.beta_up / self.beta_down * self.direction

    def computeBetadown(self):
        self.beta_down = np.sum(self.gradient ** 2)

    def computeBetaup(self):
        self.beta_up = np.sum(self.gradient ** 2)

    def computeLoss(self):
        self.loss[self.count] = np.mean((self.S.dot(self.m) - self.d) ** 2)

    def updateSoluation(self):
        self.m = self.m - self.step * self.direction

    def solve(self, epochs, error):

        self.epochs = epochs
        self.error = error
        self.count = 0

        self.loss = np.zeros(self.epochs)

        self.computeGradient()
        self.direction = copy.copy(self.gradient)

        self.computeBetaup()
        self.computeStep()

        while True:

            self.updateSoluation()
            self.computeLoss()

            self.count = self.count + 1

            print("count = ", self.count, ", loss = ", self.loss[self.count - 1])

            if self.count == self.epochs:
                break

            if self.loss[self.count] < self.error:
                break

            self.computeBetadown()
            self.computeGradient()
            self.computeBetaup()

            self.computeDirection()
            self.computeStep()

def CGsolve(sensitivity, data, model_initial, epochs, error):
    solver = CGSolver(sensitivity, data, model_initial)
    solver.solve(epochs, error)
    return solver.m 


if __name__ == "__main__":

    # set sensitivity matrix, target data and initial model
    S = np.array([[1,1,1,1,1], [1,2,2,2,2], [1,2,3,3,3], [1,2,3,4,4], [1,2,3,4,5]])
    d = np.array([1, 25, 31, -4, 5])
    m0 = np.array([1.4, -9, 23, 4, 1.9])

    # set max epochs and cut-off error
    epochs = 10
    error = 0

    def exampleSolver():
        # initial the CGsolver
        solver = CGSolver(S, d, m0)
        # solve the equation system
        solver.solve(epochs, error)
        # print the result
        result = solver.m
        print(result)

    def exampleDirectlySolve():
        result = CGsolve(S, d, m0, epochs, error)
        print(result)

    print("example 1, solve equation system by flexible solver")
    exampleSolver()

    print("example 2, solve equation system directly by a integrated function")
    exampleDirectlySolve()


