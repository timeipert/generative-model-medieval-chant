
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(matrix):
    plt.imshow(matrix, cmap='binary')
    plt.colorbar()
    plt.xlabel("to")
    plt.ylabel("from")
    plt.show()




