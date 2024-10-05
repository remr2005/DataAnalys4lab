from time import time
from math import pi, sin, cos
from dsmltf import gradient_descent, minimize_stochastic
import matplotlib.pyplot as plt

# какой то коэфициент)))
dt = 2*pi/1000
# иксы
x = list()
# ряд из 500 значений
base = [2*pi*(i/500) for i in range(500)]
# функция фурье 
def furie(k,a):
    global base
    return a[0]+a[1]*cos(base[k]) + a[2]*sin(base[k])+a[3]*cos(2*base[k])+a[4]*sin(2*base[k])

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
    global base
    x = [0,(-1)**k * dt]
    for i in range(2,500):
        x.append(x[i-1]*(2+dt*L*(1-x[i-2]**2))- x[i-2]*(1+dt**2+dt*L*(1-x[i-2]**2))+dt**2*sin(omega*dt))  
    # вычисляем коэфициенты
    s_t_0 = time()
    a0 = gradient_descent(F,[0]*5)
    s_t_1 = time()
    a1 = minimize_stochastic(f,[i for i in range(500)],[0]*500,[0]*5)
    print(a0[0],a0[1])
    print(a1[0],a1[1])
    print(f"{s_t_1-s_t_0} секунд",f"{time()-s_t_1} секунд")
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