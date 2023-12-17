#Easy iteration
import numpy as np
from math import *
from borders import u, f_main, f_test, \
                                mu1_main, mu2_main, mu3_main, mu4_main, \
                                mu1_test, mu2_test, mu3_test, mu4_test

def Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, x,y,F):
    eps_max = 0  # maximum of eps
    flag = False  

    norm_r = 0  # norm of residual
    norm_z = 0  # norm of error
    #find eigen_value and tau
    eigen_value_min = -4 * (h2 * np.sin(np.pi / (2.0 * n)) ** 2 + k2 * np.sin(np.pi / (2.0 * m)) ** 2)
    eigen_value_max = -4 * (h2 * np.cos(np.pi / (2.0 * n)) ** 2 + k2 * np.cos(np.pi / (2.0 * m)) ** 2)
    
    tau = 2 / (eigen_value_max + eigen_value_min) 
    r = np.zeros((n + 1, m + 1))  # residual

    while not flag:
        for j in range(1, m):
            for i in range(1, n):
                r[i, j] = A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j]
        eps_max = 0
        for j in range(1, m):
            for i in range(1, n):
                v_old = v[i, j]
                v_new = v_old - tau * r[i, j]
                eps_cur = abs(v_new - v_old)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v[i, j] = v_new
        counter_iterations+=1

        if eps_max < eps or counter_iterations >= N_max:
            flag = True
    for j in range(1, m):
        for i in range(1, n):
            norm_r += pow(A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j],
                          2)
            norm_z += pow(v[i][j] - u(x[i], y[j]), 2)

    norm_r = np.sqrt(norm_r)
    norm_z = np.sqrt(norm_z)
    exact = 0
    
    return(v,exact,counter_iterations, eps_max, norm_r, norm_z)