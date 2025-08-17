from . import Gauss_Points as gauss
from . import Shape_Function as sf
import numpy as np

# h = 10
# T_inf = 300
# n_c = 4
# n_c_dof = 2
# n_g_dof = 9
# gcv_x = [-1,0,1,-1,0,1,-1,0,1]
# gcv_y = [-1,-1,-1,0,0,0,1,1,1]
# C_dd =  [[3,0], [0,1], [1,2], [2,5]]

def processor_func(h, T_inf, n_c, n_c_dof, n_g_dof, gcv, C_dd):    

    # Calculation of number of gauss points
    n_g_k = int((2*(n_c_dof-1) + 1)/2)
    n_g_f = int(n_c_dof/2)
    #print(n_g_k, n_g_f)

    #Initialize global matrices
    K = np.zeros((n_g_dof,n_g_dof)) #coefficient matrix
    F = np.zeros((n_g_dof,1)) #right side vector

    #Loop over all elements
    for i in range(0, n_c):
        
        #Calculation of gauss weights and points
        gauss_e_k, gauss_w_k = gauss.weights(n_g_k)
        gauss_e_f, gauss_w_f = gauss.weights(n_g_f)

        #Initialize element matrices
        k_e = np.zeros((n_c_dof,n_c_dof)) #coefficient matrix
        f_e = np.zeros((n_c_dof,1)) #right side vector

        #Calculating length of element
        n_m = int(C_dd[i][n_c_dof - 1])
        n_1 = int(C_dd[i][0])
        
        x_m = gcv[n_m][0]; x_1 = gcv[n_1][0]; 
        y_m = gcv[n_m][1]; y_1 = gcv[n_1][1]; 
        lc = (pow((x_m - x_1),2) + pow((y_m - y_1),2))**0.5
        
        #Loop over gauss points
        for j in range(n_g_k):
            wk = gauss_w_k[j] 
            ek = gauss_e_k[j]
            
            #Calculate shape functions and derivatives
            N_e = sf.shape_vector(n_c_dof, ek)
            N_e_reshape = np.reshape(N_e, (n_c_dof))
            
            #Calculate k_e
            arr = wk*h*(lc/2)*np.matmul(N_e, np.transpose(N_e))
            k_e = k_e + arr
            
        for j in range(n_g_f):
            wk = gauss_w_f[j] 
            ek = gauss_e_f[j]
            
            #Calculate shape functions and derivatives
            N_e = sf.shape_vector(n_c_dof, ek)
            N_e_reshape = np.reshape(N_e, (n_c_dof))
            
            # Calculate f_e
            f_e = f_e + (wk*h*T_inf*lc*N_e)/2
            
        # Global assembly of K_e
        #K_e(r,s) = k_e(p,q) if C(e,p) = r && C(e,q) = s
        K_e = np.zeros((n_g_dof, n_g_dof))
        for p in range(0, n_c_dof):
            for q in range(0, n_c_dof):
                r = int(C_dd[i][p])
                s = int(C_dd[i][q])
                if r<=n_g_dof and s<=n_g_dof:
                    K_e[r][s] = k_e[p][q]

        K = K + K_e

        # Global assembly of F_e
        #F_e(r) = f_e(p) if C(e,p) = r
        F_e = np.zeros((n_g_dof,1))
        for p in range(0, n_c_dof):
            r = int(C_dd[i][p])
            if r<=n_g_dof:
                F_e[r] = f_e[p]
        
        F = F + F_e
        
    #Return necessary data
    return K, F

# processor_func(h, T_inf, n_c, n_c_dof, n_g_dof, gcv_x, gcv_y, C_dd)