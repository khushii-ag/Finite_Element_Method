#This function deals with C2 boundary with q = q* condition (specified heat flux)

from . import Gauss_Points as gauss
from . import Shape_Function as sf
import numpy as np

# q_star = 5
# n_b = 1
# n_b_dof = 2
# n_g_dof = 9
# gcv = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]
# C_d = [[8,7]] 

def processor_func(q_star, n_b, n_b_dof ,n_g_dof , gcv, C_d):

    #gcv should now store 2 columns for x,y with index representing the node number

    # Calculation of number of gauss points 
    n_g = int(n_b_dof/2)
    #print(n_g)

    #Initialize global matrices
    Q = np.zeros((n_g_dof,1)) #right side vector

    #Loop over all elements
    for i in range(0, n_b):
        
        #Calculation of gauss weights and points
        gauss_e_f, gauss_w_f = gauss.weights(n_g)
        #print("Gauss points: ",gauss_w_k,gauss_w_f)

        #Initialize element matrices
        f_e = np.zeros((n_b_dof,1)) #right side vector

        #Calculating length of element
        n_m = int(C_d[i][n_b_dof - 1]); n_1 = int(C_d[i][0])
        #will require both x,y #CHANGE
        x_m = gcv[n_m][0]; x_1 = gcv[n_1][0]
        y_m = gcv[n_m][1]; y_1 = gcv[n_1][1]
        lb = (pow((x_m - x_1),2) + pow((y_m - y_1),2))**0.5
        
        #Loop over gauss points
        for j in range(n_g):
            wk = gauss_w_f[j] 
            ek = gauss_e_f[j]
            
            #Calculate shape functions and derivatives
            N_e = sf.shape_vector(n_b_dof, ek)
            N_e_reshape = np.reshape(N_e, (n_b_dof))
            
            # Calculate f_e
            f_e = f_e - (wk*q_star*lb*N_e)/2 #wtf is this sign
             
        # Global assembly of F_e
        #F_e(r) = f_e(p) if C(e,p) = r
        Q_e = np.zeros((n_g_dof,1))
        for p in range(0, n_b_dof):
            r = int(C_d[i][p])
            if r<=n_g_dof:
                Q_e[r] = f_e[p]
        
        Q = Q + Q_e
        
    #Return necessary data
    return Q

# print(processor_func(q_star,n_b,n_b_dof ,n_g_dof ,gcv, C_d))
