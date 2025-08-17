import numpy as np

#x = (x1e + x2e)/2 + ae*e

def jacobian(ae, be):
    J = np.array([[ae/2, 0],[0, be/2]])
    return J

def t_matrix(ae, be):
    J = jacobian(ae, be)
    t_mat = np.linalg.inv(np.transpose(J))
    return t_mat