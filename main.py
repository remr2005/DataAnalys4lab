import random
from math import sqrt, pi, sin, cos
from dsmltf import gradient, gradient_descent, minimize_stochastic
import matplotlib.pyplot as plt
from statistics import mean

# какой то коэфициент)))
dt = 2*pi/1000
# иксы
x = list()
# функция фурье 
def furie(k,a):
    return a[0]+a[1]*cos(a[2]*dt*k) + a[3]*sin(a[2]*dt*k)+a[4]*cos(a[5]*dt*k)+a[6]*sin(a[5]*dt*k)

# функция ошибки для обычного градиентного спуска
def F(a:list) -> float:
    global x
    return sum([(x[j]-furie(j,a))**2 for j in range(500)])

# функция ошибки для стохатического градиентного спуска
def f(i,a):
    global x
    return (x[i]-furie(i,a))**2
 
def main():
    # какие то коэфицинты для создания ряда
    k=1
    global dt
    omega=1000/k
    L=k/100
    global x
    x = [0,(-1)**k * dt]
    for i in range(2,500):
        x.append(x[i-1]*(2+dt*L*(1-x[i-2]**2))- x[i-2]*(1+dt**2+dt*L*(1-x[i-2]**2))+dt**2*sin(omega*dt))  
    # вычисляем коэфициенты
    a0 = gradient_descent(F,[0]*7)
    a1 = minimize_stochastic(f,[i for i in range(500)],[0]*500,[0]*7)
    print(a0[0],a0[0])
    print(a1[0],a1[1])
    # ряд из 500 значений
    base = [i for i in range(500)]
    # рисуем графики
    plt.plot(base, x, label='Изначальная функция', marker='o')
    plt.plot(base, [furie(i,a0[0]) for i in range(500)], label=f'Градиентный спуск', linestyle='-')
    plt.plot(base, [furie(i,a1[0]) for i in range(500)], label=f'Стохатичный градиентный спуск', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()