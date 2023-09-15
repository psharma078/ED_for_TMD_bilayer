import bilayerTBM
import numpy as np
from numpy import sqrt, pi, cos, sin
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

ao = 3.52
theta = 4.4*pi/180
aM = ao/theta

Q_vecs = bilayerTBM.Gvecs(aM)
Q1 = Q_vecs[0]
Q2 = Q_vecs[1]

Ncutoff = 1 #choose same value from bilayer TBM
g_list =  bilayerTBM.make_g_list(Ncutoff,Q1,Q2)

Q1ortho = np.array([-Q1[1]/Q1[0],1]) #orthogonal to Q1
Q2ortho = np.array([-Q2[1]/Q2[0],1]) #orthogonal to Q2

def map_to_mBZ_momenta(q):  #q = [q] + gq
    projQ1 = np.dot(q,Q2ortho)/np.dot(Q1,Q2ortho)
    projQ2 = np.dot(q,Q1ortho)/np.dot(Q2,Q1ortho)
    IQ1 = projQ1.astype(int)
    IQ2 = projQ2.astype(int)
    FQ1 = projQ1 - IQ1
    FQ2 = projQ2 - IQ2
    gq = np.array([x*Q1 for x in IQ1]) + np.array([y*Q2 for y in IQ2])
    qmBZ = np.array([x*Q1 for x in FQ1]) + np.array([y*Q2 for y in FQ2])
    return qmBZ, gq

N = 2  #number of mBZ samples
kmesh = bilayerTBM.bz_sampling(N,Q1,Q2) #k3 and q both are kmesh

eigenSpectra = bilayerTBM.main(kmesh)
EElist = np.array([eigenSpectra[i][0] for i in range(len(kmesh))])
EVlist = np.array([eigenSpectra[i][1] for i in range(len(kmesh))])
EVlist = EVlist.reshape([len(kmesh),len(g_list),2],order='F')

q_list = np.array([k+g for k in kmesh for g in g_list])

def matching_index(arr1, arr2, tolerance=1e-6):
    abs_diff = np.abs(arr1[:, np.newaxis] - arr2)
    matching_mask = np.all(abs_diff < tolerance, axis=2)
    matching_indices = np.where(matching_mask)
    return matching_indices

def sumZstrZ(k_indxList,g_indxList):
    Nmatch = len(k_indxList[0])
    g_indxList = [g_indxList[i] for i in k_indxList[1]] #arrange glistID according to [k3+q]ID
    solve = lambda j: sum(np.sum(EVlist[k_indxList[0][j],g_indxList[j][0],:].conjugate()\
            *EVlist[k_indxList[1][j],g_indxList[j][1],:],axis=0))
    ZstrZ_sum_gl = Parallel(n_jobs=num_cores)(delayed(solve)(j) for j in range(Nmatch))
    return np.array(ZstrZ_sum_gl)    

def find_gindx(g_kq):
    g_tot_list = np.array([gkq + gp for gkq in g_kq for gp in g_list])
    g_tot_list = g_tot_list.reshape([len(g_kq),len(g_list),2])
    solver = lambda g: matching_index(g_list,g)
    gindx_list = Parallel(n_jobs=num_cores)(delayed(solver)(g) for g in g_tot_list)
    return gindx_list

def F_kgl(q):
    #computing F([k3+q],k3,g_k3+q,sigma)
    k3_plus_q_list = kmesh + q
    k3_plus_q_mBZ, g_k3q = map_to_mBZ_momenta(k3_plus_q_list)
    k1_k3_indx_list = matching_index(kmesh,k3_plus_q_mBZ)
    
    g_indx_list = find_gindx(g_k3q) #where in the g_List g_(k3+q)+g' belong to
    Zk1Zk3_list = sumZstrZ(k1_k3_indx_list,g_indx_list) #Note k1 in k1_indx and k3 in kmesh
    
    #print(sum(np.sum(EVlist[3,g_indx_list[0][0],:].conjugate()*EVlist[0,g_indx_list[0][1],:],axis=0)))
    
    #computing F([k4-q],k4,g_k4-q,sigma)
    k4_minus_q_list = kmesh - q
    k4_minus_q_mBZ, g_k4q = map_to_mBZ_momenta(k4_minus_q_list)
    k2_k4_indx_list = matching_index(kmesh,k4_minus_q_mBZ)

    g_indx_list = find_gindx(g_k4q)
    Zk2Zk4_list = sumZstrZ(k2_k4_indx_list,g_indx_list)
    #print(sum(np.sum(EVlist[3,g_indx_list[0][0],:].conjugate()*EVlist[0,g_indx_list[0][1],:],axis=0)))
    return k1_k3_indx_list, Zk1Zk3_list, k2_k4_indx_list, Zk2Zk4_list

print(F_kgl(q_list[0]))




############################################################################
"""
A = np.array([np.array([1,1]),np.array([2,2]),np.array([2,1]),np.array([1,2])])
B = np.array([np.array([1,1]),np.array([2,2]),np.array([2,1])])

print(A)
print(B)
print(np.array([x+y for y in B for x in A]))
"""
"""
if len(A)>len(B):
    r=A[:,np.newaxis,:]+B[np.newaxis,:,:]
    print(r.reshape(-1,2))
else:
    r=B[:,np.newaxis,:]+A[np.newaxis,:,:]
    print(r.reshape(-1,2))
"""

