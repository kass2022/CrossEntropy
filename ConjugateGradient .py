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
        self.beta_down = 0
        self.beta_up = 0

    def computeGradient(self, index):
        self.gradient = self.S.T.dot(self.S.dot(self.m)) - self.S.T.dot(self.d)

    def computeStep(self):
        down = np.sum((self.S.dot(self.direction)) ** 2)
        self.step = self.beta_up / down

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

    def solve(self, epochs, errors):

        self.epochs = epochs
        self.errors = errors
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

            if self.loss[self.count] < self.errors:
                break

            self.computeBetadown()
            self.computeGradient()
            self.computeBetaup()

            self.computeDirection()
            self.computeStep()




