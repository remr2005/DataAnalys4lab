def gradient(f, x, h=1e-05):
    grad = []
    for i,_ in enumerate(x):
        xh = [x_j+(h if j == i else 0) for j, x_j in enumerate(x)]
        grad.append((f(xh)-f(x))/h)
    return grad

def gradient_descent(f, x0, mu=0.05, h=1e-05, s=5000):
    x = copy(x0)
    for si in range(s):
        gf = gradient(f,x,h)
        mgf = scalar_multiply(mu*sqrt(sum(g**2 for g in gf)),gf)
        x = vector_add(x, scalar_multiply(-1, mgf))
    return x

def minimize_stochastic():
    print()

def main():
    print()

if __name__ == "__main__":
    main()