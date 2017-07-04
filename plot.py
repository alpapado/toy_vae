# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
samples = np.load("samples.npy")
x_real = np.load("real.npy")
x_hat = np.load("x_hat.npy")
plt.scatter(x=x_real[:,0], y=x_real[:,1], color='b', label='x_real'); plt.scatter(x=samples[:,0], y=samples[:,1], color='r', label='generated'); plt.scatter(x=x_hat[:,0], y=x_hat[:,1], color='g', label='x_hat'); plt.legend(loc='upper right')
plt.show()
