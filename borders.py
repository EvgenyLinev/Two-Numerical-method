#граничные условия и основные функции
import numpy as np
from math import *

# наши границы a,b,c,d= 0,2,0,1 
#задаем u и f
def u(x: float,y: float)-> float:
    return sin(pi*x*y)

def f_main(x: float,y: float)-> float:
    return -exp(-x*y**2)

def f_test(x: float,y: float)-> float:
    return sin(pi*x*y)*((pi**2)*x**2+(pi**2)*y**2)    

# задаем граничные условия
def mu1_main(y: float,a)-> float:
    return (y-2)*(y-3)
def mu2_main(y: float,b)-> float:
    return y*(y-2)*(y-3)
def mu3_main(x: float,c)-> float: 
    return (x-1)*(x-2)
def mu4_main(x: float,d)-> float: 
    return x*(x-1)*(x-2)

def mu1_test(y: float, a)-> float:
    return u(a,y)
def mu2_test(y: float, b)-> float:
    return u(b,y)
def mu3_test(x: float, c)-> float:
    return u(x,c)
def mu4_test(x: float, d)-> float:
    return u(x,d)









