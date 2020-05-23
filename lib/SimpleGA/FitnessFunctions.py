from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def multimodal(chromosome):
    x = chromosome[0]
    y = chromosome[1]

    modes = x ** 4 - 5 * x ** 2 + y ** 4 - 5 * y ** 2
    tilt = 0.5 * x * y + 0.3 * x + 15
    stretch = 0.1

    return stretch * (modes + tilt)


def plot3d(function, xinterval, yinterval):
    x = np.linspace(xinterval[0], xinterval[1], 30)
    y = np.linspace(yinterval[0], yinterval[1], 30)

    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

print("lknf")
int(0)