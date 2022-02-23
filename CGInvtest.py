from Sensitivity import GraModel
import matplotlib.pyplot as plt
from ToolsFunction import Imshow
import numpy as np
from Regularization import VariablesWeight
from ConjugateGradient import CGsolve

## forward anomaly
model = GraModel(0, 500, 35*500, 
                0, 500, 19*500, 
                0, 500, 35*500,
                0, 500, 19*500,
                0, 500, 15*500)
model.property[7:9, 9:11, 3:6] = 1
model.property[12:14, 9:11, 3:6] = 1
model.property[17:19, 9:11, 3:6] = -1
model.property[22:24, 9:11, 4:7] = -1
model.property[27:29, 9:11, 3:6] = 1
# model.property[5:15, 10:20, 5:10] = 1
# model.property[25:35, 10:20, 8:13] = -1

model.forward()

plt.figure()
Imshow(model.anomaly, "Magnetic Anomaly (nT)")

## weight the data 
S = model.sensitivity
d = model.anomaly_vector
m0 = np.zeros(model.property_vector.shape)

Variables = VariablesWeight(d, m0, S)
Variables.weight()

## Inverse the model 
epochs = 100
error = 0
Variables.mw = CGsolve(Variables.Sw, Variables.dw, Variables.mw, epochs, error)
Variables.unweight()
result = np.reshape(Variables.m,(len(model.x_model) - 1, len(model.y_model) - 1, len(model.z_model) - 1), order='F')

## show the results 
plt.figure(figsize=(10, 8))
figsize1 = 2
figsize2 = 2
slice_y = 10
i = 1
plt.subplot(figsize1,figsize2,i)
Imshow(model.property[:,slice_y,:],"original model",inverse=False)
i += 1
plt.subplot(figsize1,figsize2,i)
Imshow(result[:,slice_y,:],"result model",inverse=False)
i += 1
plt.subplot(figsize1,figsize2,i)
Imshow(model.anomaly, "original anomaly",inverse=False)
i += 1
model.property = Variables.m
model.forward()
plt.subplot(figsize1,figsize2,i)
Imshow(model.anomaly, "result anomaly",inverse=False)
plt.show()