import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class slice_graphic:

    def __init__(self, data, step):

        self.data = data
        self.step = step
        self.fig_size = (15,10)

        self.data_max = np.max(np.abs(self.data))
        self.norm = matplotlib.colors.Normalize(vmin=-self.data_max, vmax=self.data_max)

        x_len, y_len, z_len = data.shape
        self.x_index = np.linspace(0, x_len - 1, 9)
        self.y_index = np.linspace(0, y_len - 1, 9)
        self.z_index = np.linspace(0, z_len - 1, 9)

        self.xtickindex = [i for i in range(0, x_len, 5)]
        self.ytickindex = [i for i in range(0, y_len, 5)]
        self.ztickindex = [i for i in range(0, z_len, 5)]

        self.xtick = [str(i*step) for i in self.xtickindex]
        self.ytick = [str(i*step) for i in self.ytickindex]
        self.ztick = [str(i*step) for i in self.ztickindex]

        # self.FIGX = plt.figure(figsize=self.fig_size)
        # self.FIGY = plt.figure(figsize=self.fig_size)
        # self.FIGZ = plt.figure(figsize=self.fig_size)

    def coordinate_x_slice(self):

        self.FIGX = plt.figure(figsize=self.fig_size)

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            index = int(self.x_index[i])
            data_x = self.data[index, :, :]
            plt.xticks(self.ytickindex, labels=self.ytick)
            plt.yticks(self.ztickindex, labels=self.ztick)
            plt.title("Cooridinate_X = " + str(index * self.step) + "m 切片图")
            h = plt.imshow(data_x.T, cmap='bwr', norm=self.norm)
            plt.colorbar(h)

    def coordinate_y_slice(self):

        self.FIGY = plt.figure(figsize=self.fig_size)

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            index = int(self.y_index[i])
            data_y = self.data[:, index, :]
            plt.xticks(self.xtickindex, labels = self.xtick)
            plt.yticks(self.ztickindex, labels = self.ztick)
            plt.title("Cooridinate_Y = " + str(index*self.step) + "m 切片图")
            h = plt.imshow(data_y.T, cmap = 'bwr', norm = self.norm)
            plt.colorbar(h)

    def depth_z_slice(self):

        self.FIGZ = plt.figure(figsize=self.fig_size)

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            index = int(self.z_index[i])
            data_z = self.data[:, :, index]
            plt.xticks(self.xtickindex, labels = self.xtick)
            plt.yticks(self.ytickindex, labels = self.ytick)
            plt.title("Depth_Z = " + str(index*self.step) + "m 切片图")
            h = plt.imshow(data_z.T, cmap = 'bwr', norm = self.norm)
            ax.invert_yaxis()
            plt.colorbar(h)


    def presention(self):
        self.coordinate_x_slice()
        self.coordinate_y_slice()
        self.depth_z_slice()

