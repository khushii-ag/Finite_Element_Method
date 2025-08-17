# Import necessary modules

from . import Gauss_Points as gauss
from . import Shape_Function2D as sf
from . import Shape_Function_derivative as sfd
from . import Jacobian as jac
import numpy as np

def processor_func(k, r_var, n_e, n_e_dof, n_g_dof, gcv, C):

    # Calculation of number of gauss points #CHANGE
    n_g_k = int(((n_e_dof-2)*2 + 1)/2)
    n_g_f = int(n_e_dof/2)
    n_g = max(n_g_k, n_g_f)
    
    #Initialize global matrices
    K = np.zeros((n_g_dof,n_g_dof)) #coefficient matrix
    F = np.zeros((n_g_dof,1)) #right side vector

    #Loop over all elements
    for i in range(0, n_e):
        
        #Calculation of gauss weights and points
        gauss_e, gauss_w = gauss.weights(n_g)
        
        #Initialize element matrices
        k_e = np.zeros((n_e_dof,n_e_dof)) #coefficient matrix
        f_e = np.zeros((n_e_dof,1)) #right side vector

        # Calculating length and breadth of element
        n_m = int(C[i][n_e_dof - 1]); n_1 = int(C[i][0])
        x_m = gcv[n_m][0]; x_1 = gcv[n_1][0]; ae = abs(x_m - x_1)
        y_m = gcv[n_m][1]; y_1 = gcv[n_1][1]; be = abs(y_m - y_1)

        #Loop over gauss points
        for iter in range(n_g):
            wk1 = gauss_w[iter] 
            ek1 = gauss_e[iter]
        
            for j in range(n_g):
                wk2 = gauss_w[j] 
                ek2 = gauss_e[j]
                
                #Calculate shape functions and derivatives
                N_e = sf.shape_vector(n_e_dof, ek1, ek2)
                B_e = sfd.shape_vector_der(n_e_dof, ek1, ek2, ae, be)
                
                #Calculate the Jacobian
                J = jac.jacobian(ae, be)

                #Calculate k_e 
                arr = (wk1*wk2)*k*np.linalg.det(J)*np.matmul(np.transpose(B_e), B_e)
                k_e = k_e + arr
                #Calculate f_e  
                f_e = f_e + (wk1*wk2)*r_var*np.linalg.det(J)*N_e
        
        #Global assembly of K_e #CHECK
        #K_e(r,s) = k_e(p,q) if C(e,p) = r && C(e,q) = s
        K_e = np.zeros((n_g_dof, n_g_dof))
        for p in range(0, n_e_dof):
            for q in range(0, n_e_dof):
                r = int(C[i][p])
                s = int(C[i][q])
                if r<=n_g_dof and s<=n_g_dof:
                    K_e[r][s] = k_e[p][q]

        K = K + K_e

        # Global assembly of F_e
        #F_e(r) = f_e(p) if C(e,p) = r
        F_e = np.zeros((n_g_dof,1))
        for p in range(0, n_e_dof):
            r = int(C[i][p])
            if r<=n_g_dof:
                F_e[r] = f_e[p]
        
        F = F + F_e
        
    #Return necessary data
    return K, F 

# k = 1
# r_var = 1
# n_e = 1
# n_e_dof = 4
# n_g_dof = 4
# gcv = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])
# C = np.array([[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]])

# K, F = processor_func(k, r_var, n_e, n_e_dof, n_g_dof, gcv, C)
