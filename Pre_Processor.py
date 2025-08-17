import csv
import numpy as np

def pre_processor():
    rows=[]

    with open('values2.csv','r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            rows.append(row)

    f.close()

    #NODE NUMBERING STARTS FROM 0

    #n_e, n_e_dof, n_g_dof, gcv, C, q_star, n_b, n_b_dof, C_d, n_c, n_c_dof, C_dd, ess_mat
    
    # Constants
    k = float(rows[0][0])
    r_var = float(rows[1][0])
    q_star = float(rows[2][0])
    h = float(rows[3][0])
    T_inf =  float(rows[4][0])
    

    # Element properties
    n_e_x = int(rows[5][0]) #Total number of elements in x direction in domain
    n_e_y = int(rows[6][0])
    n_e = n_e_x*n_e_y
    n_b = int(rows[7][0])
    n_c = int(rows[8][0])
    n_e_dof = int(rows[9][0]) #Number of dof per element
    n_b_dof = n_c_dof = int((n_e_dof)**(1/2)) #Number of dof per boundary element


    # Creation of C
    root = int(n_e_dof**(1/2))
    r_len = root*n_e_x - (n_e_x - 1) 
    C = np.zeros((n_e_x*n_e_y, n_e_dof))
    el = 0
    for i in range(n_e_y):
        for j in range(n_e_x):
            start = j*(root - 1) + (root - 1)*(i*r_len)
            k = 0
            while k<n_e_dof:
                for l in range(root):
                    C[el][k] = int(start + l)
                    k += 1
                start += r_len
            el += 1

    n_g_dof = int(C[el-1][k-1] + 1)

    # Global Coordinate Vector
    gcv = [] #Global coordinate vector - X coord, Y coord at node number
    for i in range(0, len(rows[10]), 2):
        gcv.append([float(rows[10][i]), float(rows[10][i+1])])

    # C'
    C_d = np.zeros((n_b, n_b_dof))
    for i in range(0, n_b):
        for j in range(0, n_b_dof):
            C_d[i][j] = int(rows[11 + i][j])

    # C''
    C_dd = np.zeros((n_c,n_c_dof))
    for i in range(0, n_c):
        for j in range(0, n_c_dof):
            C_dd[i][j] = int(rows[11 + n_b + i][j])

    # Essential Boundary Condition Matrix
    ess_mat = []
    for i in range(0, len(rows[11 + n_b + n_c]), 2):
        ess_mat.append([int(rows[11+ n_b + n_c][i]),int(rows[11 + n_b + n_c][i+1])]) #Node number, value
    
    return k, r_var, n_e, n_e_dof, n_g_dof, gcv, C, q_star, n_b, n_b_dof, C_d, h, T_inf, n_c, n_c_dof, C_dd, ess_mat


pre_processor()