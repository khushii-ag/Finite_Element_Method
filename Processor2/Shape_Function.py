#Lagrangian Shape function construction
import numpy as np

def shape_vector(n_e_dof, e):

    #Generate natural coordinate node vector
    ep = np.linspace(-1, 1, num = n_e_dof)
    arr_n = np.zeros((n_e_dof,1))

    #Iterate over each element in node vector
    for i in range(0, n_e_dof):
        numerator = 1

        # Calculate numerator of Lagrangian shape function
        for j in range(0,n_e_dof):
            if(i!=j): numerator*=(e - ep[j])

        denominator = 1

        # Calculate denominator of Lagrangian shape function
        for j in range(0, n_e_dof):
            if(i!=j): denominator*=(ep[i] - ep[j])

        # Calculate Lagrangian shape function value for each node
        arr_n[i] = numerator/denominator

    arr_n.shape = (n_e_dof,1)
    return arr_n

