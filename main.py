import random
from math import sqrt

def gradient(f, x, h=1e-05):
    grad = []
    for i,_ in enumerate(x):
        xh = [x_j+(h if j == i else 0) for j, x_j in enumerate(x)]
        grad.append((f(xh)-f(x))/h)
    return grad

def scalar_multiply(n,x):
    return [n*i for i in x]

def vector_add(a,b):
    return [a[i]+b[i] for i in range(len(a))]

def gradient_descent(f, x0, mu=0.05, h=1e-05, s=5000):
    x = x0.copy()
    for si in range(s):
        gf = gradient(f,x,h)
        mgf = scalar_multiply(mu*sqrt(sum(g**2 for g in gf)),gf)
        x = vector_add(x, scalar_multiply(-1, mgf))
    return x

# Стохастический градиентный спуск
def minimize_stochastic(f, x, y, a_0, h_0 = 0.1, max_steps = 1000):
    a = a_0
    h = h_0
    min_a, min_F = None, float('inf')
    drunken_steps = 0
    while drunken_steps < max_steps:
        value = sum((f(xx,a)-yy)**2 for xx,yy in zip(x,y))
        if value < min_F:
            min_a, min_F = a, value
            drunken_steps = 0
            h = h_0
        else:
            drunken_steps += 1
            h *= 0.9
            n = random.randint(0,len(x)-1)
            grad = []
            for i,_ in enumerate(a):
                ah = a.copy()
                ah[i] += h
                # главное - в этом шаге (суммы нет!)
                grad.append(((f(x[n],ah)-y[n])**2-(f(x[n],a)-y[n])**2)/h)
    a = [a[i] - h*grad[i] for i,_ in enumerate(a)]
    return min_a, value

def main():
    print()

if __name__ == "__main__":
    main()