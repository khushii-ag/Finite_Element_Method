import Pre_Processor as pre
from Processor1 import Processor1
from Processor2 import Processor2
from Processor3 import Processor3
import numpy as np


def super_processor():

    k, r_var, n_e, n_e_dof, n_g_dof, gcv, C, q_star, n_b, n_b_dof, C_d, h, T_inf, n_c, n_c_dof, C_dd, ess_mat = pre.pre_processor()

    K, Q = Processor1.processor_func(k, r_var, n_e, n_e_dof, n_g_dof, gcv, C)
    Qd = Processor2.processor_func(q_star, n_b, n_b_dof, n_g_dof, gcv, C_d)
    Kdd, Qdd = Processor3.processor_func(h, T_inf, n_c, n_c_dof, n_g_dof, gcv, C_dd) 

    global_K = K + Kdd 
    global_Q = Q + Qd + Qdd

    #Putting essential boundary condition
    for i in ess_mat: #Here i is of the form [node, val = T_star]
        subtract = global_K[i[0]][:]*i[1]
        subtract.shape = (len(global_K),1)
        global_Q = global_Q - subtract
        global_K[:,i[0]] = 0
        global_K[i[0],:] = 0
        global_K[i[0]][i[0]] = 1
        global_Q[i[0]] = i[1]

    #Solving the system of equations
    # [K]{u} = {F}
    prim_var = np.linalg.solve(global_K, global_Q)
    prim_var.shape = (n_g_dof,1)
    #print(prim_var)

    return prim_var, gcv

# k=2
# r_var=2
# n_e=4
# n_e_dof=4
# n_g_dof=9
# gcv = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])
# C = np.array([[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]])
# q_star=2
# n_b=2
# n_b_dof=2
# C_d=np.array([[3,0],[0,1]])
# h=2
# T_inf=10
# n_c=2
# n_c_dof=2
# C_dd=np.array([[1,2],[2,5]])
# ess_mat=np.array([[6,2],[7,2],[8,2]])

# super_processor(k, r_var, n_e, n_e_dof, n_g_dof, gcv, C, q_star, n_b, n_b_dof, C_d, h, T_inf, n_c, n_c_dof, C_dd, ess_mat)