import numpy as np
from numpy import sin, cos

pi = 3.1415926535
u0 = 4*pi*10**(-7)

class MagModel:

    def __init__(self, x_measure_start, x_measure_step, x_measure_end,
                y_measure_start, y_measure_step, y_measure_end,
                x_model_start, x_model_step, x_model_end,
                y_model_start, y_model_step, y_model_end,
                z_model_start, z_model_step, z_model_end):

        self.x_measure = np.arange(x_measure_start, x_measure_end + 1, x_measure_step)
        self.y_measure = np.arange(y_measure_start, y_measure_end + 1, y_measure_step)
        self.x_model = np.arange(x_model_start, x_model_end + 1, x_model_step)
        self.y_model = np.arange(y_model_start, y_model_end + 1, y_model_step)
        self.z_model = np.arange(z_model_start, z_model_end + 1, z_model_step)
        self.z_model = self.z_model + 0.1

        self.property = np.zeros((len(self.x_model) - 1, len(self.y_model) - 1, len(self.z_model) - 1))

        self.measurenum = len(self.x_measure)*len(self.y_measure)
        self.modelnum = (len(self.x_model)-1)*(len(self.y_model)-1)*(len(self.z_model)-1)

        self.iM = 3.1415926535*(90/180)
        self.dM = 3.1415926535*(90/180)

        self.xz_weight = - cos(self.iM) * cos(self.dM)
        self.yz_weight = - cos(self.iM) * sin(self.dM)
        self.zz_weight = sin(self.iM)

    def computeSensitivity(self):

        M = np.zeros(
            (len(self.y_measure), len(self.x_measure), len(self.x_model), len(self.y_model), len(self.z_model)))

        X, Y = np.meshgrid(self.x_measure, self.y_measure)

        for i in range(len(self.x_model)):
            for j in range(len(self.y_model)):
                for k in range(len(self.z_model)):
                    r = np.sqrt((self.x_model[i] - X) ** 2 + (self.y_model[j] - Y) ** 2 + self.z_model[k] ** 2)
                    M[:, :, i, j, k] = self.zz_weight * np.arctan(- (X - self.x_model[i]) * (Y - self.y_model[j]) / r / self.z_model[k])

        A = M[:, :, 1:, 1:, 1:] \
             + M[:, :, 0:-1, 0:-1, 1:] \
             + M[:, :, 0:-1, 1:, 0:-1] \
             + M[:, :, 1:, 0:-1, 0:-1] \
             - M[:, :, 0:-1, 1:, 1:] \
             - M[:, :, 1:, 0:-1, 1:] \
             - M[:, :, 1:, 1:, 0:-1] \
             - M[:, :, 0:-1, 0:-1, 0:-1]

        A = 10**9*u0*A/4/pi

        self.sensitivity = np.reshape(A, (self.measurenum, self.modelnum), order='F')

    def forward(self):

        self.computeSensitivity()
        self.property_vector = np.reshape(self.property, (self.modelnum,), order='F')
        self.anomaly_vector = self.sensitivity.dot(self.property_vector)
        self.anomaly = np.reshape(self.anomaly_vector, (len(self.x_measure), len(self.y_measure)))

def magSensitivity(x_measure_start, x_measure_step, x_measure_end,
                    y_measure_start, y_measure_step, y_measure_end,
                    x_model_start, x_model_step, x_model_end,
                    y_model_start, y_model_step, y_model_end,
                    z_model_start, z_model_step, z_model_end):

    model = MagModel(x_measure_start, x_measure_step, x_measure_end,
                    y_measure_start, y_measure_step, y_measure_end,
                    x_model_start, x_model_step, x_model_end,
                    y_model_start, y_model_step, y_model_end,
                    z_model_start, z_model_step, z_model_end)

    model.computeSensitivity()

    return model.sensitivity


class GraModel:

    def __init__(self, x_measure_start, x_measure_step, x_measure_end,
                y_measure_start, y_measure_step, y_measure_end,
                x_model_start, x_model_step, x_model_end,
                y_model_start, y_model_step, y_model_end,
                z_model_start, z_model_step, z_model_end):

        self.x_measure = np.arange(x_measure_start, x_measure_end + 1, x_measure_step)
        self.y_measure = np.arange(y_measure_start, y_measure_end + 1, y_measure_step)
        self.x_model = np.arange(x_model_start, x_model_end + 1, x_model_step)
        self.y_model = np.arange(y_model_start, y_model_end + 1, y_model_step)
        self.z_model = np.arange(z_model_start, z_model_end + 1, z_model_step)
        self.z_model = self.z_model + 0.1

        self.property = np.zeros((len(self.x_model) - 1, len(self.y_model) - 1, len(self.z_model) - 1))

        self.measurenum = len(self.x_measure)*len(self.y_measure)
        self.modelnum = (len(self.x_model)-1)*(len(self.y_model)-1)*(len(self.z_model)-1)

    def computeSensitivity(self):

        M = np.zeros((len(self.y_measure),len(self.x_measure), len(self.x_model), len(self.y_model), len(self.z_model)))

        X, Y = np.meshgrid(self.x_measure, self.y_measure)

        for i in range(len(self.x_model)):
            for j in range(len(self.y_model)):
                for k in range(len(self.z_model)):
                    r = np.sqrt((self.x_model[i] - X)**2 + (self.y_model[j] - Y)**2 + self.z_model[k]**2)

                    M[:, :, i, j, k] = (self.x_model[i] - X) * np.log(r + Y - self.y_model[j]) \
                                       + (self.y_model[j] - Y) * np.log(r + X - self.x_model[i]) \
                                       - self.z_model[k] * np.arctan( - (X - self.x_model[i]) * (Y - self.y_model[j]) / r / self.z_model[k])

        A = M[:, :, 1:, 1:, 1:] \
             + M[:, :, 0:-1, 0:-1, 1:] \
             + M[:, :, 0:-1, 1:, 0:-1] \
             + M[:, :, 1:, 0:-1, 0:-1] \
             - M[:, :, 0:-1, 1:, 1:] \
             - M[:, :, 1:, 0:-1, 1:] \
             - M[:, :, 1:, 1:, 0:-1] \
             - M[:, :, 0:-1, 0:-1, 0:-1]

        A = A * 6.67e-3

        self.sensitivity = np.reshape(A , (self.measurenum, self.modelnum), order = 'F')

    def forward(self):

        self.computeSensitivity()
        self.property_vector = np.reshape(self.property, (self.modelnum,), order='F')
        self.anomaly_vector = self.sensitivity.dot(self.property_vector)
        self.anomaly = np.reshape(self.anomaly_vector, (len(self.x_measure), len(self.y_measure)))

def graSensitivity(x_measure_start, x_measure_step, x_measure_end,
                    y_measure_start, y_measure_step, y_measure_end,
                    x_model_start, x_model_step, x_model_end,
                    y_model_start, y_model_step, y_model_end,
                    z_model_start, z_model_step, z_model_end):

    model = GraModel(x_measure_start, x_measure_step, x_measure_end,
                    y_measure_start, y_measure_step, y_measure_end,
                    x_model_start, x_model_step, x_model_end,
                    y_model_start, y_model_step, y_model_end,
                    z_model_start, z_model_step, z_model_end)
    model.computeSensitivity()

    return model.sensitivity


if __name__ == "__main__":

    from ToolsFunction import Imshow
    import matplotlib.pyplot as plt

    def exampleForward():

        # define the 3D grids of the model 
        # MagModel is used to forward magnetic anomaly, if you want to forward gravity, use GraModel
        model = MagModel(0,100,1000, 
                        0,100,700, #start, step, and end location of measure system(m)
                        0,100,1000,
                        0,100,700,
                        0,100,500)#start, step, and end location of underground model(m)

        # set the properties of underground space, here we set a cuboid model(A/m in mag, g/cm^3 in gra)
        model.property[3:5, 2:4, 2:4] = 1

        # compute anomaly(nT in mag, mGal in gra) 
        model.forward()

        # show the results
        plt.figure()
        Imshow(model.anomaly, "anomaly")


    def exampleSensitivity():

        # compute the sensitivity of model directly, often used in inverse
        # graSensitivity and magSensitivity function are available
        sensitivity = graSensitivity(0,100,1000,
                                    0,100,700, #start, step, and end location of measure system(m)
                                    0,100,1000,
                                    0,100,700,
                                    0,100,500)#start, step, and end location of underground model(m)

        print(sensitivity)
        print(sensitivity.shape)
        

    exampleForward()
    exampleSensitivity()
    plt.show()
