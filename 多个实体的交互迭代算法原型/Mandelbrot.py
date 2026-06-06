import numpy as np
import matplotlib.pyplot as plt

def f1(z):
    return z * z

def f2(c):
    return c

def h(a, b):
    return a + b

class Parameters:
    def __init__(self, max_iter=256, W1_init=0+0j, W2_init=0+0j):
        self.max_iter = max_iter
        self.W1_init = W1_init
        self.W2_init = W2_init

def g(f1, f2, h, P):
    W1 = P.W1_init 
    W2 = P.W2_init 
    for n in range(P.max_iter):
        a = f1(W1)
        b = f2(W2)
        W1 = h(a, b)
        if abs(W1) > 2:
            return n
    return P.max_iter

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    P = Parameters(max_iter=max_iter, W1_init=0+0j, W2_init=0+0j)
    escape_times = np.zeros((height, width))

    for i in range(height):
        y = ymax - i * (ymax - ymin) / (height - 1)
        for j in range(width):
            x = xmin + j * (xmax - xmin) / (width - 1)
            c = complex(x, y)

            P.W2_init = c

            escape_times[i, j] = g(f1, f2, h, P)

    return escape_times

if __name__ == "__main__":
    WIDTH, HEIGHT = 800, 600
    MAX_ITER = 256
    XMIN, XMAX, YMIN, YMAX = -2.0, 1.0, -1.2, 1.2

    escape = generate_mandelbrot(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)

    plt.figure(figsize=(10, 7))
    plt.imshow(escape, extent=[XMIN, XMAX, YMIN, YMAX], cmap='hot', origin='lower')
    plt.colorbar(label='逃逸迭代次数')
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()
