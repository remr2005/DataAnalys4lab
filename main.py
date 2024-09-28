import random
from math import sqrt, pi, sin, cos
from dsmltf import gradient, gradient_descent, minimize_stochastic
import matplotlib.pyplot as plt

x = list()

def furie(k,a):
    return a[0] + a[1]*cos(a[2]*k) + a[3]* sin(k) + a[3]*cos(2*k) + a[4]*sin(2*k)

def F(a:list) -> float:
    global x
    return sum([abs(x[j]-furie(j,a)) for j in range(500)])

def f(i,a):
    global x
    return abs(x[i]-furie(i,a))

def main():
    k=1
    dt=2*pi/1000
    omega=1000/k
    L=k/100
    global x
    x = [0,(-1)**k * dt]
    for i in range(2,500):
        x.append(x[i-1]*(2+dt*L*(1-x[i-2]**2))- x[i-2]*(1+dt**2+dt*L*(1-x[i-2]**2))+dt**2*sin(omega*dt))  
    a0 = gradient_descent(F,[0,0,0,0,0])
    a1 = minimize_stochastic(f,[i for i in range(500)],[0]*500,[0,0,0,0,0])
    print(a0[0],a1[0])
    print(a0[1],a1[1])
    base = [i for i in range(500)]
    plt.plot(base, x, label='Изначальная функция', marker='o')
    plt.plot(base, [furie(i,a0[0]) for i in range(500)], label=f'Градиентный спуск', linestyle='-')
    plt.plot(base, [furie(i,a1[0]) for i in range(500)], label=f'Стохатичный градиентный спуск', linestyle='-')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.title('Аппроксимация полиномом')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()