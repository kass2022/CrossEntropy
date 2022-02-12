import numpy as np

class CG:

    ## d = Am
    ## objective function: ||Am-d||2  +  alpha*||m||2

    def __init__(self, SensitivityMatrix, TargetData, InitialModel, Wm):

        self.A = SensitivityMatrix
        self.d = TargetData
        self.m = InitialModel

        self.m_unweight = InitialModel
        self.Wm = Wm

        self.gradient = None
        self.constrained_gradient = None
        self.constrained_gradient = None
        self.cross_gradient = None

        self.direction = None
        self.step = 0
        self.beta_down = 0
        self.beta_up = 0

    def compute_gradient(self, index):

        def original_gradient():
            self.gradient = self.A.T.dot(self.A.dot(self.m)) - self.A.T.dot(self.d)

        def constrained_gradient(index):

            if index == 'L2':
                print(np.mean(self.gradient))
                self.constrained_gradient = 1 / (self.Wm)**2

                p1 = np.sum((self.A.dot(self.m) - self.d)**2)
                p2 = np.sum(self.m**2)*self.alpha

                print('p1 = ', p1, ', p2 = ', p2 )
                self.constrained_gradient *= self.alpha * p1 / (p2+0.1)
                print(np.mean(self.constrained_gradient*self.m))
                self.gradient += self.constrained_gradient * self.m

            if index == 'L0':
                a =  1 / self.Wm**2
                self.constrained_gradient =  self.alpha * ( a *1e-2 / (a*self.m**2 + 1e-2)**2) * self.Wm**2
                self.gradient += self.constrained_gradient * self.m

        original_gradient()
        constrained_gradient(index)

    # def cross_entropy_gradient(self, cross_gradient):
    #     self.cross_gradient = self.lambd * cross_gradient / (self.m + 0.0001)
    #     self.gradient += self.cross_gradient * (self.m + 0.0001)

    def compute_step(self):

        down = np.sum((self.A.dot(self.direction)) ** 2)
        down += np.sum(self.constrained_gradient * self.direction ** 2)
        # down += np.sum(self.cross_gradient * self.direction ** 2)
        self.step = self.beta_up / down

    # def compute_step_cross(self):
    #
    #     down = np.sum((self.A.dot(self.direction)) ** 2)
    #     down += np.sum(self.constrained_gradient * self.direction ** 2)
    #     down += np.sum(self.cross_gradient * self.direction ** 2)
    #     self.step = self.beta_up / down

    def compute_direction(self):
        self.direction = self.gradient + self.beta_up / self.beta_down * self.direction

    def compute_beta_down(self):
        self.beta_down = np.sum(self.gradient ** 2)

    def compute_beta_up(self):
        self.beta_up = np.sum(self.gradient ** 2)

    def compute_loss(self):
        self.loss[self.count] = np.mean((self.A.dot(self.m) - self.d) ** 2)

    def update_soluation(self):
        self.m = self.m - self.step * self.direction

    def iteration(self, index, alpha, epochs, errors, cross_entropy):

        self.epochs = epochs
        self.errors = errors
        self.alpha = alpha
        self.count = 0

        self.loss = np.zeros(self.epochs)

        self.compute_gradient(index)
        self.direction = self.gradient

        self.compute_beta_up()
        self.compute_step()

        while True:

            self.update_soluation()
            self.compute_loss()

            self.count = self.count + 1

            print("count = ", self.count, ", loss = ", self.loss[self.count - 1])

            # print(self.step)
            # print(self.step * self.Wm**2)
            if self.count == self.epochs:
                break

            if self.loss[self.count] < self.errors:
                break

            self.compute_beta_down()
            self.compute_gradient(index)
            self.compute_beta_up()

            self.compute_direction()
            self.compute_step()

    # def iteration1(self, index, alpha, lambd, epochs, errors, cross_entropy):
    #
    #     self.epochs = epochs
    #     self.errors = errors
    #     self.alpha = alpha
    #     self.lambd = lambd
    #     self.count = 0
    #
    #     self.loss = np.zeros(self.epochs)
    #
    #     self.compute_beta_down()
    #     self.compute_gradient(index)
    #     self.cross_entropy_gradient(cross_entropy)
    #     self.compute_beta_up()
    #     self.compute_direction()
    #     self.compute_step_cross()
    #
    #     self.update_soluation()
    #     self.compute_loss()
    #     print(self.loss)





