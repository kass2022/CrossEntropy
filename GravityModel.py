import numpy as np

G = 6.67259e-11

class data():
    def __init__(self):
        self.dg = None
        self.Gxx = None
        self.Gyy = None
        self.Gxy = None
        self.Gxz = None
        self.Gyz = None
        self.Gzz = None

class underground():
    
    def __init__(self, x_start, x_step, x_end, y_start, y_step, y_end, z_start, z_step, z_end):
        
        self.x = np.arange(x_start, x_end+1, x_step)
        self.y = np.arange(y_start, y_end+1, y_step)
        self.z = np.arange(z_start, z_end+1, z_step)
        self.x -= np.min(self.x)
        self.y -= np.min(self.y)
        self.z -= np.min(self.z)
        self.z = self.z + 0.1
        self.x = self.x + 0.1
        self.y = self.y + 0.1
        
        self.measurenum = len(self.x)*len(self.y)
        self.modelnum = (len(self.x)-1)*(len(self.y)-1)*(len(self.z)-1)

        self.property = None
        self.property_vector = None
        self.anomaly = data()
        self.anomaly_vector = data()
        self.A = data()

        self.property = np.zeros((len(self.x) - 1, len(self.y) - 1, len(self.z) - 1))

    def Compute_sensitivity_matrix(self, Index):

        def chose():

            if Index == 'dg':
                def core_fun(r, i, j, k):
                    result = self.dg_fun(r, i, j, k)
                    return result
                para1 = 1
                para2 = 1
                return core_fun, para1, para2

            if Index == 'Gxx':
                def core_fun(r, i, j, k):
                    result = self.Gxx_fun(r, i, j, k)
                    return result
                para1 = 1
                para2 = 1
                return core_fun, para1, para2

            if Index == 'Gyy':
                def core_fun(r, i, j, k):
                    result = self.Gyy_fun(r, i, j, k)
                    return result
                para1 = 1
                para2 = 1
                return core_fun, para1, para2

            if Index == 'Gxy':
                def core_fun(r, i, j, k):
                    result = self.Gxy_fun(r, i, j, k)
                    return result
                para1 = -1
                para2 = -1
                return core_fun, para1, para2

            if Index == 'Gxz':
                def core_fun(r, i, j, k):
                    result = self.Gxz_fun(r, i, j, k)
                    return result
                para1 = 1
                para2 = -1
                return core_fun, para1, para2

            if Index == 'Gyz':
                def core_fun(r, i, j, k):
                    result = self.Gyz_fun(r, i, j, k)
                    return result
                para1 = -1
                para2 = 1
                return core_fun, para1, para2

            if Index == 'Gzz':
                def core_fun(r, i, j, k):
                    result = self.Gzz_fun(r, i, j, k)
                    return result
                para1 = 1
                para2 = 1
                return core_fun, para1, para2

        def expend(A_single, para1, para2):

            fliplr = np.fliplr(A_single)
            M_fliplr = np.concatenate((para1*fliplr, A_single), axis = 1)
            del fliplr, A_single

            flipud = np.flipud(M_fliplr)
            M = np.concatenate((para2*flipud, M_fliplr), axis = 0)

            return M

        def compute_M(core_fun, para1, para2):

            A_point = np.empty((len(self.x), len(self.y), len(self.z)), dtype=np.float64)

            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    for k in range(len(self.z)):
                        r = np.sqrt(self.x[i] ** 2 + self.y[j] ** 2 + self.z[k] ** 2)
                        A_point[i, j, k] = core_fun(r, i, j, k)

            A_single \
                = A_point[1:, 1:, 1:] \
                  + A_point[0:-1, 0:-1, 1:] \
                  + A_point[0:-1, 1:, 0:-1] \
                  + A_point[1:, 0:-1, 0:-1] \
                  - A_point[0:-1, 1:, 1:] \
                  - A_point[1:, 0:-1, 1:] \
                  - A_point[1:, 1:, 0:-1] \
                  - A_point[0:-1, 0:-1, 0:-1]

            M = expend(A_single, para1, para2)

            return M

        def extract_A(M):

            xmid = len(self.x) - 1
            ymid = len(self.y) - 1
            xlen = 2 * xmid
            ylen = 2 * ymid

            count = 0

            A =  np.zeros(( (len(self.x)-1) * (len(self.y)-1) * (len(self.z)-1), len(self.x) * (len(self.y)) ))

            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    Target = M[(xmid - i):(xlen - i), (ymid - j):(ylen - j), :]
                    A_column = np.reshape(Target, [(len(self.x) - 1) * (len(self.y) - 1) * (len(self.z) - 1), 1], order='F')
                    A[:, count] = A_column[:, 0]
                    count += 1

            return A.T

        def main():

            core_fun, para1, para2 = chose()
            M = compute_M(core_fun, para1, para2)
            A = extract_A(M)

            if Index == 'dg':
                self.A.dg = A * G*1e8
            if Index == 'Gxx':
                self.A.Gxx = A * G*1e12
            if Index == 'Gyy':
                self.A.Gyy = A * G*1e12
            if Index == 'Gxy':
                self.A.Gxy = A * G*1e12
            if Index == 'Gxz':
                self.A.Gxz = A * G*1e12
            if Index == 'Gyz':
                self.A.Gyz = A * G*1e12
            if Index == 'Gzz':
                self.A.Gzz = A * G*1e12

        try:
            main()
        except:
            print('can not compute sensitivity matrix')

    def dg_fun(self, r, i, j, k):
        result = self.x[i] * np.log(r - self.y[j]) \
                 + self.y[j] * np.log(r - self.x[i])\
                 - self.z[k] * np.arctan(- self.x[i] * self.y[j] / r / self.z[k])
        return result

    def Gxx_fun(self, r, i, j, k):
        return np.arctan( - self.y[j] * self.z[k] / self.x[i] / r)

    def Gyy_fun(self, r, i ,j, k):
        return np.arctan( - self.x[i] *  self.z[k] / self.y[j] / r)

    def Gxy_fun(self, r, i, j, k):
        return  np.log( r - self.z[k])

    def Gxz_fun(self, r, i, j, k):
        return - np.log(r - self.y[j])

    def Gyz_fun(self, r, i, j, k):
        return  np.log(r - self.x[i])

    def Gzz_fun(self, r, i, j, k):
        return np.arctan( - self.x[i] * self.y[j] / self.z[k] / r)

    def forward(self, Index):

        self.Compute_sensitivity_matrix(Index)
        self.property_vector = np.reshape(self.property, (self.modelnum,), order='F')

        if Index == 'dg':
            self.anomaly_vector.dg = self.A.dg.dot(self.property_vector)
            self.anomaly.dg = np.reshape(self.anomaly_vector.dg, (len(self.x), len(self.y)))
        if Index == 'Gxx':
            self.anomaly_vector.Gxx = self.A.Gxx.dot(self.property_vector)
            self.anomaly.Gxx = np.reshape(self.anomaly_vector.Gxx, (len(self.x), len(self.y)))
        if Index == 'Gyy':
            self.anomaly_vector.Gyy = self.A.Gyy.dot(self.property_vector)
            self.anomaly.Gyy = np.reshape(self.anomaly_vector.Gyy, (len(self.x), len(self.y)))
        if Index == 'Gxy':
            self.anomaly_vector.Gxy = self.A.Gxy.dot(self.property_vector)
            self.anomaly.Gxy = np.reshape(self.anomaly_vector.Gxy, (len(self.x), len(self.y)))
        if Index == 'Gxz':
            self.anomaly_vector.Gxz = self.A.Gxz.dot(self.property_vector)
            self.anomaly.Gxz = np.reshape(self.anomaly_vector.Gxz, (len(self.x), len(self.y)))
        if Index == 'Gyz':
            self.anomaly_vector.Gyz = self.A.Gyz.dot(self.property_vector)
            self.anomaly.Gyz = np.reshape(self.anomaly_vector.Gyz, (len(self.x), len(self.y)))
        if Index == 'Gzz':
            self.anomaly_vector.Gzz = self.A.Gzz.dot(self.property_vector)
            self.anomaly.Gzz = np.reshape(self.anomaly_vector.Gzz, (len(self.x), len(self.y)))


