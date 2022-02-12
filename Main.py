import GravityModel
import MagneticModel
import CrossEntropy
import matplotlib.pyplot as plt
import numpy as np
import WeightMatrix
from Inversion import CG
from graphic import slice_graphic

# iM=3.1415926535*(90/180)
# dM=(3.1415926535*(90-(0))/180)
#
# model_mag = MagneticModel.underground(0,1000,20001,0,1000,20001,0,1000,20001,iM,dM )
# model_mag.property[7:13,7:13,5:10] = 1
# model_mag.forward()
#
# m_m = model_mag.property_vector
# A_m = model_mag.A.dT
# d_m = model_mag.anomaly_vector.dT

model_gra = GravityModel.underground(0,100,2300,0,100,1600,0,100,1000)
model_gra.property[6:10:,6:10,2:5] = 1
model_gra.property[14:18:,6:10,2:5] = 1
# model_gra.property[6:10:,6:10,2:5] = 1

model_gra.forward('dg')

m_g = model_gra.property_vector
A_g = model_gra.A.dg
d_g = model_gra.anomaly_vector.dg

wg = WeightMatrix.weight(m_g, d_g, A_g)
wg.weighting('Zhdanov', 'Zhdanov')

# wm = WeightMatrix.weight(m_m, d_m, A_m)
# wm.weighting('Zhdanov', 'Zhdanov')

gra_m0 = np.zeros(model_gra.modelnum)
A = wg.Aw

d = wg.dw
A = 1*A#/np.max(d)
d = d#/np.max(d)

# gra_inv = CG(wg.Aw, d_g, gra_m0, Wm = wg.solution_weighting.weight)
gra_inv = CG(A, d, gra_m0, Wm = wg.solution_weighting.weight)
gra_inv.iteration(index = 'L2', alpha=1e-2, epochs=100, errors=0, cross_entropy=0)
# gra_inv = CG(wg.Aw, wg.dw, gra_inv.m, Wm = wg.solution_weighting.weight)
# gra_inv.iteration(index = 'L0', alpha=1e-4, epochs=10000, errors=0, cross_entropy=0)
# gra_m0 = np.ones(model_gra.modelnum)
# # gra_inv = CG(wg.Aw, d_g, gra_m0, Wm = wg.solution_weighting.weight)
# gra_inv = CG(A_g, d_g, gra_m0, Wm = wg.solution_weighting.weight)
# gra_inv.iteration(index = 'L2', alpha=0, epochs=3, errors=0, cross_entropy=0)

# mag_m0 = np.zeros(model_mag.modelnum)
# mag_inv = CG(wm.Aw, wm.dw, mag_m0, Wm = wm.solution_weighting.weight)
# mag_inv.iteration(index = 'L2', alpha=1, epochs=1, errors=0, cross_entropy=0)

# for i in range(50):
#
#     t = CrossEntropy.CrossEntropy(np.reshape(gra_inv.m/wg.solution_weighting.weight, [20, 20, 20], order = 'F'),
#                                   np.reshape(mag_inv.m/wm.solution_weighting.weight, [20, 20, 20], order = 'F'),
#                                   1, 1, 1)
#
#     t.compute_gradient_of_CrossEntropy()
#     gra_cross_gradient = np.reshape(t.property1.gradient, [8000,], order = 'F')
#     mag_cross_gradient = np.reshape(t.property2.gradient, [8000,], order = 'F')
#
#     gra_inv.iteration1(index = 'L0', alpha=1, lambd = 1e5,epochs=1, errors=0, cross_entropy = gra_cross_gradient)
#     mag_inv.iteration1(index = 'L0', alpha=1, lambd = 1e6,epochs=1, errors=0, cross_entropy = mag_cross_gradient)
#
#     print(i)
def present_dg():
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(model_gra.anomaly.dg, cmap='rainbow')
    plt.colorbar()

    wg.solution_weighting.unweighting(gra_inv.m)
    rho = wg.solution_weighting.solution

    # rho[rho < 0.1] = 0
    # rho = gra_inv.m

    model_pre = GravityModel.underground(0,100,2300,0,100,1600,0,100,1000)
    model_pre.property = np.reshape(rho, (23,16,10), order='F')
    model_pre.forward('dg')
    plt.subplot(1,3,2)
    plt.imshow(model_pre.anomaly.dg,cmap='rainbow')
    plt.colorbar()

    err = model_pre.anomaly.dg - model_gra.anomaly.dg
    plt.subplot(1,3,3)
    plt.imshow(err,cmap='bwr')
    plt.colorbar()
    rho = model_pre.property
    figure = slice_graphic(rho, 100)
    figure.presention()
    figure2 = slice_graphic(model_gra.property, 100)
    figure2.presention()

def present_mag():

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(model_mag.anomaly.dT, cmap='rainbow')
    plt.colorbar()

    wm.solution_weighting.unweighting(mag_inv.m)

    mag = wm.solution_weighting.solution

    model_pre = MagneticModel.underground(0,1000,20001,0,1000,20001,0,1000,20001,iM,dM)
    model_pre.property = np.reshape(mag, (20,20,20), order='F')
    model_pre.forward()
    plt.subplot(1,3,2)
    plt.imshow(model_pre.anomaly.dT,cmap='rainbow')
    plt.colorbar()

    err = model_pre.anomaly.dT - model_mag.anomaly.dT
    plt.subplot(1,3,3)
    plt.imshow(err,cmap='bwr')
    plt.colorbar()

    mag = model_pre.property
    figure2 = slice_graphic(mag, 100)
    figure2.presention()
# t = CrossEntropy.CrossEntropy(model_gra.property, model_mag.property, 1, 1, 1)
# t.compute_CrossEntropy()
# t.compute_gradient_of_CrossEntropy()

def present_cross():

    m1 = np.reshape(gra_inv.m / wg.solution_weighting.weight, [20, 20, 20], order='F')
    m2 = np.reshape(mag_inv.m / wm.solution_weighting.weight, [20, 20, 20], order='F')
    plt.figure(figsize=(10,20))

    plt.subplot(2,2,1)
    plt.imshow(m1[10, :, :].T, cmap = 'bwr')
    plt.title('density')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(m2[10, :, :].T, cmap = 'bwr')
    plt.title('magnetic')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(t.property1.gradient[10, :, :].T, cmap = 'bwr')
    plt.title('density gradient of cross entropy')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(t.property2.gradient[10, :, :].T, cmap = 'bwr')
    plt.title('magnetic gradient of cross entropy')
    plt.colorbar()

    plt.figure(figsize=(10,20))
    plt.subplot(2,2,1)
    plt.imshow(t.cross_entropy.x[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(t.cross_entropy.y[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(t.cross_entropy.z[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(t.cross_entropy.whole[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.suptitle('cross entropy')

    plt.figure(figsize=(10,20))
    plt.subplot(2,2,1)
    plt.imshow(t.property1.directional.x[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(t.property1.directional.y[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(t.property1.directional.z[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.suptitle('density directional')

    plt.figure(figsize=(10,20))
    plt.subplot(2,2,1)
    plt.imshow(t.property2.directional.x[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(t.property2.directional.y[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(t.property2.directional.z[10, :, :].T, cmap = 'bwr')
    plt.colorbar()
    plt.suptitle('magnrtic directional')

present_dg()
# present_cross()
# present_mag()
# plt.savefig()
plt.show()
