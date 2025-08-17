import numpy as np

def L(n_e_dof, num, e, coords):
    numerator = 1

    # Calculate numerator of Lagrangian shape function
    for j in range(0,n_e_dof):
        if(num!=j): numerator*=(e - coords[j])

    denominator = 1

    # Calculate denominator of Lagrangian shape function
    for j in range(0, n_e_dof):
        if(num!=j): denominator*=(coords[num] - coords[j])
    
    # Calculate Lagrangian shape function value
    arr_n = numerator/denominator

    return arr_n

def shape_vector(p, e, n):

    N = np.zeros((p, 1))
    dof_b = int(p**(1/2))
    coords = np.linspace(-1, 1, num = dof_b)
    
    count = 0
    for i in range(dof_b): #n loop
        for j in range(dof_b): #e loop
            N[count] = L(dof_b, i, n, coords)*L(dof_b, j, e, coords) #remember i,j start at 0
            count+=1

    return N
