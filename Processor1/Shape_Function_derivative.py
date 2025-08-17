import numpy as np
from . import Shape_Function2D as sf
from . import Jacobian as jac

def der_L(n_e_dof, num, e, coords):
    
    #Numerical differentiation
    h = 1e-6
    N_h = sf.L(n_e_dof, num, e+h, coords) #(n_e_dof, e+h)
    N = sf.L(n_e_dof, num, e-h, coords) #(n_e_dof, e-h)

    B = (N_h - N)/(2*h)
    return B


def shape_vector_der(p, e, n, ae, be): #0 based indexing

    B = np.zeros((2, p))
    dof_b = int(p**(1/2))
    coords = np.linspace(-1, 1, num = dof_b)
    
    #derivative wrt to e
    count = 0
    for i in range(dof_b): #n loop
        for j in range(dof_b): #e loop
            B[0][count] = sf.L(dof_b, i, n, coords)*der_L(dof_b, j, e, coords)
            count += 1

    #derivative wrt to n
    count = 0
    for i in range(dof_b): #n loop
        for j in range(dof_b): #e loop
            B[1][count] = der_L(dof_b, i, n, coords)*sf.L(dof_b, j, e, coords)
            count += 1
    
    #conversion from master system to local system
    B_conv = np.matmul(jac.t_matrix(ae, be), B)

    return B_conv
