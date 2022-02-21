import numpy as np

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

    def computeDirectional(self):

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

    def computeCrossEntropy(self):

        self.property1.computeDirectional()
        self.property2.computeDirectional()

        self.cross_entropy.x = self.property1.directional.y * self.property2.directional.z \
            - self.property1.directional.z * self.property2.directional.y
        self.cross_entropy.y = self.property1.directional.x * self.property2.directional.z \
            - self.property1.directional.z * self.property2.directional.x
        self.cross_entropy.z = self.property1.directional.x * self.property2.directional.y \
            - self.property1.directional.y * self.property2.directional.x        
        self.cross_entropy.whole = self.cross_entropy.x**2 + self.cross_entropy.y**2 + self.cross_entropy.z**2

    def computeGradient(self):

        def computeGradientIJK(data1,data2,i,j,k):

            temp_gradient = self.cross_entropy.y[i,j,k]*data2.directional.z[i,j,k]/self.x_step \
                + self.cross_entropy.z[i,j,k]*data2.directional.y[i,j,k]/self.x_step
            if temp_gradient!= 0:
                data1.gradient[i - 1, j, k] -= temp_gradient
                data1.gradient[i + 1, j, k] += temp_gradient

            temp_gradient = self.cross_entropy.x[i,j,k]*data2.directional.z[i,j,k]/self.y_step\
                +self.cross_entropy.z[i,j,k]*data2.directional.x[i,j,k]/self.y_step
            if temp_gradient != 0:
                data1.gradient[i,j-1,k] -= temp_gradient
                data1.gradient[i,j+1,k] += temp_gradient   

            temp_gradient = self.cross_entropy.x[i,j,k]*data2.directional.y[i,j,k]/self.z_step\
                -self.cross_entropy.y[i,j,k]*data2.directional.x[i,j,k]/self.z_step   
            if temp_gradient!= 0:
                data1.gradient[i,j,k-1] -= temp_gradient
                data1.gradient[i,j,k+1] += temp_gradient   

        self.computeCrossEntropy()

        for i in range(1, self.x_num - 1):
            for j in range(1, self.y_num - 1):
                for k in range(1, self.z_num - 1):
                    computeGradientIJK(self.property1, self.property2, i, j, k)

if __name__ == "__main__":

    # define 2 models of different properties : m1 and m2
    x_num = 100
    y_num = 100
    z_num = 100

    m1 = np.zeros((x_num,y_num,z_num))
    m1[40:60, 30:70, 20:80] = 1

    m2 = np.zeros((x_num,y_num,z_num))
    for i in range(x_num):
        m2[:,:,i] = i

    # compute crossentropy of m1 and m2, also its gradient
    x_step = 1000
    y_step = 1000
    z_step = 1000
    t = CrossEntropy(m1, m2, x_step, y_step, z_step)
    t.computeGradient()    

    # show the results
    
    import matplotlib.pyplot as plt

    plt.figure()
    figsize1 = 3
    figsize2 = 3
    slice_y = 50
    i = 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(m1[:,slice_y,:].T)
    plt.colorbar()
    plt.title("m1")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(m2[:,slice_y,:].T)
    plt.colorbar()
    plt.title("m2")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.x[:,slice_y,:].T)
    plt.colorbar()
    plt.title("tx")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.y[:,slice_y,:].T)
    plt.colorbar()
    plt.title("ty")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.z[:,slice_y,:].T)
    plt.colorbar()
    plt.title("tz")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.whole[:,slice_y,:].T)
    plt.colorbar()
    plt.title("t")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.property1.gradient[:,slice_y,:].T)
    plt.colorbar()
    plt.title("t_gradient")

    plt.suptitle("y = 50")

    plt.figure()
    figsize1 = 3
    figsize2 = 3
    slice_z = 50
    i = 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(m1[:,:,slice_z].T)
    plt.colorbar()
    plt.title("m1")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(m2[:,:,slice_z].T)
    plt.colorbar()
    plt.title("m2")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.x[:,:,slice_z].T)
    plt.colorbar()
    plt.title("tx")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.y[:,:,slice_z].T)
    plt.colorbar()
    plt.title("ty")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.z[:,:,slice_z].T)
    plt.colorbar()
    plt.title("tz")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.cross_entropy.whole[:,:,slice_z].T)
    plt.colorbar()
    plt.title("t")
    i += 1
    plt.subplot(figsize1,figsize2,i)
    plt.imshow(t.property1.gradient[:,:,slice_z].T)
    plt.colorbar()
    plt.title("t_gradient")

    plt.suptitle("z = 50")

    plt.show()

    
          

