import numpy as np
from GravityModel import underground

class directional():
    def __init__(self, shape):
        self.x = np.zeros(shape)
        self.y = np.zeros(shape)
        self.z = np.zeros(shape)
        self.whole = np.zeros(shape)

class data():

    def __init__(self, property, x_step, y_step, z_step ):

        self.property = property
        self.x_num, self.y_num, self.z_num = self.property.shape
        self.x_step = x_step
        self.y_step = y_step
        self.z_step = z_step

        self.directional = directional(self.property.shape)
        self.gradient = np.zeros(self.property.shape)

    def compute_directional(self):

        for i in range(1, self.x_num - 1):
            for j in range(1, self.y_num - 1):
                for k in range(1, self.z_num - 1):

                    self.directional.x[i, j, k] = self.property[i - 1, j, k] - self.property[i + 1, j, k]
                    self.directional.y[i, j, k] = self.property[i, j - 1, k] - self.property[i, j + 1, k]
                    self.directional.z[i, j, k] = self.property[i, j, k - 1] - self.property[i, j, k + 1]

        self.directional.x /= 2 * self.x_step
        self.directional.y /= 2 * self.y_step
        self.directional.z /= 2 * self.z_step

class CrossEntropy():

    def __init__(self, property1, property2, x_step, y_step, z_step):

        self.x_step = x_step
        self.y_step = y_step
        self.z_step = z_step

        self.property1 = data(property1, x_step, y_step, z_step)
        self.property2 = data(property2, x_step, y_step, z_step)
        self.cross_entropy = directional(self.property1.property.shape)
        self.t = np.zeros(self.property1.property.shape)

        self.x_num, self.y_num, self.z_num = self.property1.property.shape

    def compute_CrossEntropy(self):

        self.property1.compute_directional()
        self.property2.compute_directional()

        self.cross_entropy.x = self.property1.directional.y * self.property2.directional.z - self.property1.directional.z * self.property2.directional.y
        self.cross_entropy.y = self.property1.directional.x * self.property2.directional.z - self.property1.directional.z * self.property2.directional.x
        self.cross_entropy.z = self.property1.directional.x * self.property2.directional.y - self.property1.directional.y * self.property2.directional.x        
        self.cross_entropy.whole = self.cross_entropy.x**2 + self.cross_entropy.y**2 + self.cross_entropy.z**2

    # def compute_gradient_of_CrossEntropy(self):

    #     def core_fun(data1, data2, i, j, k):

    #         temp_gradient = (data2.directional.y[i, j, k] * self.cross_entropy.x[i, j, k] \
    #             + data2.directional.z[i, j, k] * self.cross_entropy.y[i, j, k]) / self.x_step
    #         data1.gradient[i - 1, j, k] += temp_gradient
    #         data1.gradient[i + 1, j, k] -= temp_gradient

    #         temp_gradient = (data2.directional.x[i, j, k] * self.cross_entropy.z[i, j, k] \
    #             + data2.directional.z[i, j, k] * self.cross_entropy.x[i, j, k]) / self.y_step
    #         data1.gradient[i, j - 1, k] += temp_gradient
    #         data1.gradient[i, j + 1, k] -= temp_gradient

    #         temp_gradient = (data2.directional.x[i, j, k] * self.cross_entropy.y[i, j, k] \
    #             + data2.directional.y[i, j, k] * self.cross_entropy.x[i, j, k]) / self.z_step
    #         data1.gradient[i, j, k - 1] += temp_gradient
    #         data1.gradient[i, j, k + 1] -= temp_gradient

    #     for i in range(1, self.x_num - 1):
    #         for j in range(1, self.y_num - 1):
    #             for k in range(1, self.z_num - 1):
    #                 core_fun(self.property1, self.property2, i, j, k)
    #                 core_fun(self.property2, self.property1, i, j, k)


    def compute_gradient_of_CrossEntropy(self):

        self.compute_CrossEntropy()

        def gradient_of_tx(data1, data2, i, j, k):
            
            if self.cross_entropy.x[i, j, k] != 0:

                gradient = np.zeros(self.property1.property.shape)

                temp_gradient = data2.directional.z[i, j, k] / 2 * self.y_step
                gradient[i, j - 1, k] = + temp_gradient
                gradient[i, j + 1, k] = - temp_gradient

                temp_gradient = data2.directional.y[i, j, k] / 2 * self.z_step
                gradient[i, j, k - 1] = - temp_gradient
                gradient[i, j, k + 1] = + temp_gradient

                data1.gradient += gradient * self.cross_entropy.x[i, j, k]

        def gradient_of_ty(data1, data2, i, j, k):
            
            if self.cross_entropy.y[i, j, k] != 0:

                gradient = np.zeros(self.property1.property.shape)

                temp_gradient = data2.directional.z[i, j, k] / 2 * self.x_step
                gradient[i - 1, j, k] = + temp_gradient
                gradient[i + 1, j, k] = - temp_gradient

                temp_gradient = data2.directional.x[i, j, k] / 2 * self.z_step
                gradient[i, j, k - 1] = - temp_gradient
                gradient[i, j, k + 1] = + temp_gradient

                data1.gradient += gradient * self.cross_entropy.y[i, j, k]

        def gradient_of_tz(data1, data2, i, j, k):

            if self.cross_entropy.z[i, j, k] != 0:
                
                gradient = np.zeros(self.property1.property.shape)

                temp_gradient = data2.directional.y[i, j, k] / 2 * self.x_step
                gradient[i - 1, j, k] = + temp_gradient
                gradient[i + 1, j, k] = - temp_gradient

                temp_gradient = data2.directional.x[i, j, k] / 2 * self.y_step
                gradient[i, j - 1, k] = - temp_gradient
                gradient[i, j + 1, k] = + temp_gradient

                data1.gradient += gradient * self.cross_entropy.z[i, j, k]

        for i in range(1, self.x_num - 1):
            for j in range(1, self.y_num - 1):
                for k in range(1, self.z_num - 1):

                    gradient_of_tx(self.property1, self.property2, i, j, k)
                    gradient_of_tx(self.property2, self.property1, i, j, k)
                    gradient_of_ty(self.property1, self.property2, i, j, k)
                    gradient_of_ty(self.property2, self.property1, i, j, k)
                    gradient_of_tz(self.property1, self.property2, i, j, k)
                    gradient_of_tz(self.property2, self.property1, i, j, k)
                
                

