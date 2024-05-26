import matplotlib.pyplot as plt
import numpy as np
def plot_function_values(*args):
    plt.figure()
    for values, label in args:
        plt.plot(values, label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Values over Iterations')
    plt.legend()
    plt.show()

def plot_contour_with_paths(func, limits, paths=[]):
    x = np.linspace(limits[0], limits[1], 100)
    y = np.linspace(limits[2], limits[3], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    plt.figure()
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    for path, label in paths:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], marker='o', label=label)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour plot with optimization paths')
    plt.legend()
    plt.show()
