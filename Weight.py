from matplotlib.pyplot import magnitude_spectrum
from numpy import zeros, sqrt, sum
import copy

class VariablesWeight:

    def __init__(self, data, model, sensitivity):

        self.d = data
        self.dw = zeros(len(self.d))     
        self.m = model
        self.mw = zeros(len(self.m))        
        self.S = sensitivity
        self.Sw = zeros(self.S.shape)

        self.Wd = zeros(len(self.d))
        self.Wm = zeros(len(self.m))

    def zhdanovWd(self):
        for i in range(self.S.shape[0]):
            self.Wd[i] = sqrt(sum(self.S[i, :] ** 2))

    def zhdanovWm(self):
        for i in range(self.S.shape[1]):
            self.Wm[i] = sqrt(sum(self.S[:, i] ** 2))

    def weightData(self):
        self.dw = self.d * self.Wd

    def weightModel(self):        
        self.mw = self.m * self.Wm

    def weightSensitivity(self):
        for i in range(self.S.shape[0]):
            self.Sw[i,:] = self.S[i,:] * self. Wd[i]
        for i in range(self.S.shape[1]):
            self.Sw[:,i] = self.Sw[:,i] / self.Wm[i]

    def weight(self, data_flag = 'zhdanov', model_flag = 'zhdanov'):

        if data_flag == 'zhdanov':
            self.zhdanovWd()

        if model_flag == 'zhdanov':
            self.zhdanovWm()

        self.weightData()
        self.weightModel()
        self.weightSensitivity()
 
    def unweight(self):
        self.m = self.mw / self.Wm


if __name__ == "__main__":

    from Sensitivity import MagModel
    import matplotlib.pyplot as plt
    from ToolsFunction import Imshow

    # define a model and compute the anomaly and sensitivity
    model = MagModel(0, 100, 300, 0, 100, 300, 0, 100, 300, 0, 100, 300, 0, 100, 300)
    model.property[1,:,1] = 1
    model.property[0,:,0] = -1
    model.forward()

    # define variables used for test weight 
    S = model.sensitivity
    d = model.anomaly_vector
    m = model.property_vector

    # weight variables 
    Variables = VariablesWeight(d, m, S)
    Variables.weight()

    # show the results
    print("d:", Variables.d)
    print("dw:", Variables.dw)
    print("m:", Variables.m)
    print("mw:", Variables.mw)

    plt.figure(figsize=(10,8))
    figsize1 = 1
    figsize2 = 2

    i = 1
    plt.subplot(figsize1,figsize2,i)
    Imshow(Variables.S, 'S', 'bwr')
    i += 1
    plt.subplot(figsize1,figsize2,i)
    Imshow(Variables.Sw, 'Sw', 'bwr')

    plt.suptitle("the response between sensitivity and\n \
         underground spaces of different depth\n \
         is more uniform after weighting")

    plt.show()




