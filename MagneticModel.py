import numpy as np
from numpy import sin, cos

pi = 3.1415926535
u0=4*pi*10**(-7)

class data():
    def __init__(self):
        self.dT = None
        self.Tx = None
        self.Ty = None
        self.Tz = None
        self.Bxx = None
        self.Byy = None
        self.Bxy = None
        self.Bxz = None
        self.Byz = None
        self.Bzz = None

class underground():

    def __init__(self, x_start, x_step, x_end, y_start, y_step, y_end, z_start, z_step, z_end, iM, dM):

        self.x = np.arange(x_start, x_end + 1, x_step)
        self.y = np.arange(y_start, y_end + 1, y_step)
        self.z = np.arange(z_start, z_end + 1, z_step)
        self.x -= np.min(self.x)
        self.y -= np.min(self.y)
        self.z -= np.min(self.z)
        self.z = self.z + 0.1
        self.x = self.x + 0.1
        self.y = self.y + 0.1

        self.iM = iM
        self.dM = dM
        self.compute_weight()

        self.measurenum = len(self.x) * len(self.y)
        self.modelnum = (len(self.x) - 1) * (len(self.y) - 1) * (len(self.z) - 1)

        self.property = None
        self.property_vector = None
        self.anomaly = data()
        self.anomaly_vector = data()
        self.A = data()

        self.property = np.zeros((len(self.x) - 1, len(self.y) - 1, len(self.z) - 1))

    def compute_weight(self):
        self.xz_weight = - cos(self.iM) * cos(self.dM)
        self.yz_weight = - cos(self.iM) * sin(self.dM)
        self.zz_weight = sin(self.iM)

    def Compute_sensitivity_matrix(self):

        def expend(A_single):

            fliplr = np.fliplr(A_single)
            M_fliplr = np.concatenate((fliplr, A_single), axis=1)
            del fliplr, A_single

            flipud = np.flipud(M_fliplr)
            M = np.concatenate((flipud, M_fliplr), axis=0)

            return M

        def compute_M():

            A_point = np.empty((len(self.x), len(self.y), len(self.z)), dtype=np.float64)

            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    for k in range(len(self.z)):
                        r = np.sqrt(self.x[i] ** 2 + self.y[j] ** 2 + self.z[k] ** 2)
                        A_point[i, j, k] = self.dT_fun(r, i, j, k)

            A_single \
                = A_point[1:, 1:, 1:] \
                  + A_point[0:-1, 0:-1, 1:] \
                  + A_point[0:-1, 1:, 0:-1] \
                  + A_point[1:, 0:-1, 0:-1] \
                  - A_point[0:-1, 1:, 1:] \
                  - A_point[1:, 0:-1, 1:] \
                  - A_point[1:, 1:, 0:-1] \
                  - A_point[0:-1, 0:-1, 0:-1]

            M = expend(A_single)

            return M

        def extract_A(M):

            xmid = len(self.x) - 1
            ymid = len(self.y) - 1
            xlen = 2 * xmid
            ylen = 2 * ymid

            count = 0

            A = np.zeros(((len(self.x) - 1) * (len(self.y) - 1) * (len(self.z) - 1), len(self.x) * (len(self.y))))

            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    Target = M[(xmid - i):(xlen - i), (ymid - j):(ylen - j), :]
                    A_column = np.reshape(Target, [(len(self.x) - 1) * (len(self.y) - 1) * (len(self.z) - 1), 1],
                                          order='F')
                    A[:, count] = A_column[:, 0]
                    count += 1

            return A.T

        def main():

            M = compute_M()
            A = extract_A(M)
            self.A.dT = 10**9*u0*A/4/pi

        main()

    def dT_fun(self, r, i, j, k):
        result = self.xz_weight * np.log(r - self.y[j]) \
                 + self.yz_weight * np.log(r - self.x[i]) \
                 + self.zz_weight * np.arctan(- self.x[i] * self.y[j] / r / self.z[k])
        return result


    def forward(self):

        self.Compute_sensitivity_matrix()
        self.property_vector = np.reshape(self.property, (self.modelnum,), order='F')

        self.anomaly_vector.dT = self.A.dT.dot(self.property_vector)
        self.anomaly.dT = np.reshape(self.anomaly_vector.dT, (len(self.x), len(self.y)))


