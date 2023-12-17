#grid_calculation
import numpy as np
from math import *
from Chebishev import Chebishev_metod
from Easy_iteration import Simple_metod
from borders import u

def start_main(n, m, N_max, eps, a, b, c, d, Var, F, mu1, mu2, mu3, mu4, k_num):
    h = (b - a) / (n)  
    k = (d - c) / (m)  
    x = np.linspace(a, b, n + 1)  
    y = np.linspace(c, d, m + 1)  
    f = np.array([np.array([F(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # f(x, y)
    mu_1 = np.array([mu1(i,a) for i in y])  
    mu_2 = np.array([mu2(i,b) for i in y])  
    mu_3 = np.array([mu3(i,c) for i in x])  
    mu_4 = np.array([mu4(i,d) for i in x])  
    
    v = np.zeros((n + 1, m + 1))  
    
    counter_steps = 0  
    counter_iterations = 0  
    h2 = 1 / h ** 2  
    k2 = 1 / k ** 2  
    A = -2 * (h2 + k2)  
    
    v[0, :] = mu_1  
    v[n, :] = mu_2  
    v[:, 0] = mu_3  
    v[:, m] = mu_4  

    for j in range(1, m):
        v[1:n, j] = mu_1[j] + (mu_2[j] - mu_1[j]) / (b - a) * (x[1:n] - a)

    R0=0
    for i in range(len(f)):
        for j in range(len(f[i])):
            if R0<abs(v[i,j]-f[i,j]):
                R0=abs(v[i,j]-f[i,j])

    if Var == 1:
        v_method, exact, counter_iterations, eps_max, norm_r, norm_z=\
            Chebishev_metod(h2, k2, n, m, k_num, A, v, f, eps, N_max, counter_iterations, counter_steps, x, y, F)
    elif Var == 2:
        v_method, exact, counter_iterations, eps_max, norm_r, norm_z=\
            Simple_metod(h2, k2, n, m, k_num, A, v, f, eps, N_max, counter_iterations, x, y, F)

    ### теперь половинный шаг
    n_half = n*2
    m_half = m*2
    h_half = (b - a) / (n_half)  
    k_half = (d - c) / (m_half)  
    x_half = np.linspace(a, b, n_half + 1)  
    y_half = np.linspace(c, d, m_half + 1)  
    f_half = np.array([np.array([F(x_half[i], y_half[j]) for j in range(m_half + 1)]) for i in range(n_half + 1)])  # f(x, y)
    mu_1_half = np.array([mu1(i,a) for i in y_half])  
    mu_2_half = np.array([mu2(i,b) for i in y_half])  
    mu_3_half = np.array([mu3(i,c) for i in x_half])  
    mu_4_half = np.array([mu4(i,d) for i in x_half])  
    
    v_half = np.zeros((n_half + 1, m_half + 1))  
    
    counter_steps_half = 0  
    counter_iterations_half = 0  
    h2_half = 1 / h_half ** 2  
    k2_half = 1 / k_half ** 2  
    A_half = -2 * (h2_half + k2_half)  
    
    v_half[0, :] = mu_1_half 
    v_half[n_half, :] = mu_2_half  
    v_half[:, 0] = mu_3_half  
    v_half[:, m_half] = mu_4_half  
    # linear interpolation by x
    for j in range(1, m_half):
        v_half[1:n_half, j] = mu_1_half[j] + (mu_2_half[j] - mu_1_half[j]) / (b - a) * (x_half[1:n_half] - a)

    if Var == 1:
        v_method_half, exact_half, counter_iterations_half, eps_max_half, norm_r_half, norm_z_half = \
            Chebishev_metod(h2_half, k2_half, n_half, m_half, k_num, A_half,
                            v_half, f_half, eps, N_max, counter_iterations_half, 
                            counter_steps_half, x_half, y_half, F)
    elif Var == 2:
        v_method_half, exact_half, counter_iterations_half, eps_max_half, norm_r_half, norm_z_half = \
            Simple_metod(h2_half, k2_half, n_half, m_half, k_num, A_half, v_half, f_half,
                         eps, N_max, counter_iterations_half, x_half, y_half, F)

    v_half_out = np.zeros((int(n) + 1, int(m) + 1))
    for i in range(0, n*2 + 1, 2):
        for j in range(0, m*2 + 1, 2):
            v_half_out[int(i / 2), int(j / 2)] = v_method_half[i, j]
    ### половинный шаг окончен 

    v=[[0 for i in range (n + 1)] for j in range(m+1)]
    for i in range(m + 1):
        for j in range(n + 1):
            v[j][i] = round(v_method[j][i], 15)
    
    return v, v_half_out, abs(v-v_half_out), x, y, counter_iterations, counter_iterations_half, eps_max, norm_r, norm_z, R0
    
def start_test(n, m, N_max, eps, a, b, c, d, Var, F, mu1, mu2, mu3, mu4, k_num):
      
    h = (b - a) / (n)  
    k = (d - c) / (m)  
    x = np.linspace(a, b, n + 1)  
    y = np.linspace(c, d, m + 1)  
    f = np.array([np.array([F(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # f(x, y)
    mu_1 = np.array([mu1(i,a) for i in y])  
    mu_2 = np.array([mu2(i,b) for i in y])  
    mu_3 = np.array([mu3(i,c) for i in x])  
    mu_4 = np.array([mu4(i,d) for i in x])  
    
    v = np.zeros((n + 1, m + 1))  
    
    counter_steps = 0  
    counter_iterations = 0  
    h2 = 1 / h ** 2  
    k2 = 1 / k ** 2  
    A = -2 * (h2 + k2)  
    
    v[0, :] = mu_1  
    v[n, :] = mu_2  
    v[:, 0] = mu_3  
    v[:, m] = mu_4  
    
    # linear interpolation by x
    for j in range(1, m):
        v[1:n, j] = mu_1[j] + (mu_2[j] - mu_1[j]) / (b - a) * (x[1:n] - a)
    
    #невязка на начальном приближении
    R0 = 0
    for i in range(len(f)): 
        for j in range(len(f[i])):
            if R0<abs(v[i,j] - u(i,j)):
                R0=abs(v[i,j] - u(i,j))
    
    if Var == 1:
        v_method, exact, counter_iterations, eps_max, norm_r, norm_z = \
            Chebishev_metod(h2, k2, n, m, k_num, A, v, f, eps, 
                            N_max, counter_iterations, counter_steps, x, y, F)
    elif Var==2:
        v_method, exact, counter_iterations, eps_max, norm_r, norm_z = \
            Simple_metod(h2, k2, n, m, k_num, A, v, f, eps,
                          N_max, counter_iterations, x, y, F)

    v=[[0 for i in range (n+1)] for j in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            v[j][i] = round(v_method[j][i], 12)
    
    exact = np.array([np.array([u(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # u(x, y)

    return v, exact, abs(v - exact), x, y, counter_iterations, eps_max, norm_r, norm_z, R0