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


def rosenbrock(chromosome, a, b):
    x = chromosome[0]
    y = chromosome[1]

    return (a-x)**2+b*(y-x**2)**2


def plot3d(function, xinterval, yinterval):
    x = np.linspace(xinterval[0], xinterval[1], 100)
    y = np.linspace(yinterval[0], yinterval[1], 100)

    X, Y = np.meshgrid(x, y)
    Z = function([X, Y], a=1, b=100)

    fig = plt.figure(figsize=plt.figaspect(0.4), dpi=300)
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    fig.suptitle(r'Function $f(x,y)=0.1*((x^4-5x^2+y^4-5y^2)+0.5xy+0.3x+15)$')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis_r', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$f(x,y)$')
    ax.view_init(55, -135)
    # ax.view_init(25, -35)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    p1 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis_r', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zticks([])
    ax.view_init(90, -135)
    # ax.view_init(90, 35)

    fig.colorbar(p1, shrink=0.5, aspect=10)

    """

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(80, 0)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(25, 135)"""

    plt.show()


if __name__ == "__main__":
    plot3d(rosenbrock, [-2, 2], [-1, 3])
