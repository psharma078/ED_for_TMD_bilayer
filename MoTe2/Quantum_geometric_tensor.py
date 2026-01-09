import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import sin,cos,exp, pi, sqrt,log
from joblib import Parallel, delayed
import multiprocessing
from itertools import combinations, permutations, repeat
from numba import jit,njit
from numba.typed import List
from pylanczos import PyLanczos
import time
import sys

num_cores = multiprocessing.cpu_count()

def main(theta,Vd):
    theta = theta/180*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2

    Nk = 6 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3
    n_g = (2*Nk+1)**2
    Nklist = np.arange(-Nk,Nk+1)
    Nkoneslist = np.ones(2*Nk+1)
    mmlist = np.kron(Nklist,Nkoneslist)
    nnlist = np.kron(Nkoneslist,Nklist)
    
    Qxlist = nnlist*Q1[0] + mmlist*Q2[0]
    Qylist = nnlist*Q1[1] + mmlist*Q2[1]
    #print(np.vstack((nnlist,mmlist)).T)
    diagm = np.diag(np.ones(2*Nk,dtype=np.complex128),-1)
    diagp = diagm.T
    diag_Nk = np.eye(2*Nk+1,dtype=np.complex128)
    delta_nn_mmp1 = np.kron(diagm, diag_Nk)
    delta_nnp1_mm = np.kron(diag_Nk,diagm)
    delta_nnm1_mmm1 = np.kron(diagp,diagp)
    delta_nn_mm = np.kron(diag_Nk,diag_Nk)
    delta_nn_mmm1 = delta_nn_mmp1.T

    halfdim = (Nk*2+1)**2
    ham_dim = 2*halfdim
    ### kline Gamma -- M ---K---Gamma
    def special_kline(npts):
        Gammapt = np.array([0.,0.])
        mpt = Q1/2
        kline0 = klines(Gammapt, kappa_p, npts)
        kline1 = klines(kappa_p,mpt,npts)
        kline2 = klines(mpt,Gammapt,npts)
        kline3 = klines(Gammapt,kappa_m, npts)
        klabels = [r"$\gamma$", r'$\kappa_+$', r'$m$', r'$\gamma$', r'$\kappa_-$']
        kcoord = np.array([0,(npts-1),2*(npts-1),3*(npts-1)+1,4*(npts-1)+1])
        return np.concatenate((kline0[:-1,:],kline1[:-1,:],kline2[:-1,:],kline3[:,:])), kcoord, klabels

    def gen_layer_hamiltonian(k):
        me = 9.10938356e-31 # kg
        m = 0.62*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV
        kxlist_top = Qxlist + k[0] - kappa_p[0]
        kylist_top = Qylist + k[1] - kappa_p[1]
        kxlist_bottom = Qxlist + k[0] - kappa_m[0]
        kylist_bottom = Qylist + k[1] - kappa_m[1]
        # First add potential terms
        psi = -91/180.0*pi
        V = 11.2 # meV

        H_top = -V*(delta_nn_mmp1 + delta_nnp1_mm + delta_nnm1_mmm1)*exp(1j*psi)
        H_top += H_top.conj().T
        H_bottom = H_top.conj()
        # Now add kinetic terms
        np.fill_diagonal(H_top,(kxlist_top**2 + kylist_top**2)*prefactor)
        np.fill_diagonal(H_bottom,(kxlist_bottom**2 + kylist_bottom**2)*prefactor)

        #Add displacement field
        np.fill_diagonal(H_top, H_top.diagonal() + Vd/2.0)
        np.fill_diagonal(H_bottom, H_bottom.diagonal() - Vd/2.0)
        return H_top, H_bottom

    def H_der(k):
        me = 9.10938356e-31 # kg
        m = 0.62*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV
        kxlist_top = Qxlist + k[0] - kappa_p[0]
        kylist_top = Qylist + k[1] - kappa_p[1]
        kxlist_bottom = Qxlist + k[0] - kappa_m[0]
        kylist_bottom = Qylist + k[1] - kappa_m[1]
        ham_x = np.zeros((ham_dim,ham_dim),dtype=np.complex128)
        ham_y = np.zeros((ham_dim,ham_dim),dtype=np.complex128)
        np.fill_diagonal(ham_x[:halfdim, :halfdim], 2*prefactor*kxlist_top)
        np.fill_diagonal(ham_y[:halfdim, :halfdim], 2*prefactor*kylist_top)
        np.fill_diagonal(ham_x[halfdim:, halfdim:], 2*prefactor*kxlist_bottom)
        np.fill_diagonal(ham_y[halfdim:, halfdim:], 2*prefactor*kylist_bottom)
        return ham_x, ham_y

    def mat_ele_H_der(k, psi0, psi1):
        ham_x, ham_y = H_der(k)
        return psi0.conj()@ham_x@psi1, psi0.conj()@ham_y@psi1

    def gen_tunneling():
        # w = -18.0 # meV
        w = 13.3
        Delta_T = -w*(delta_nn_mm + delta_nnm1_mmm1 + delta_nn_mmm1)
        return Delta_T

    def construct_ham(k):
        H_top,H_bottom = gen_layer_hamiltonian(k)
        Delta_T = gen_tunneling()
        ham = np.zeros((ham_dim,ham_dim),dtype=np.complex128)
        ham[:halfdim,:halfdim] = H_top
        ham[halfdim:,halfdim:] = H_bottom
        ham[:halfdim,halfdim:] = Delta_T.conj().T
        ham[halfdim:,:halfdim] = Delta_T#.conj().T
        return ham

    def quantum_metric_tensor(HH,Q1,Q2,dim,Vd):
        npts = 16
        k1list = np.linspace(-0.5,0.5,npts+1)
        k2list = np.linspace(-0.5,0.5,npts+1)
        # Diagonalizing Hamiltonian at each k
        k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(npts+1) for ii in range(npts+1)])
        kxylist = k12list @ np.array([Q1,Q2])
        result = parallel_diag([HH(k) for k in kxylist])
        kxylist = kxylist.reshape((npts+1,npts+1,2))
        EEs = np.array([result[jj][0] for jj in range(len(result))]).reshape((npts+1,npts+1,dim))
        Evecs = np.array([result[jj][1] for jj in range(len(result))]).reshape((npts+1,npts+1,dim,dim))
        del result
        k_area = np.linalg.norm(Q1) * np.linalg.norm(Q2) * sqrt(3)/2/(npts**2)
    
        trace_g = 0.0
        for ii in range(npts):
            for jj in range(npts):
                mat_x, mat_y = mat_ele_H_der(kxylist[ii,jj,:], Evecs[ii,jj,:,-1], Evecs[ii,jj,:,:-1])
                trace_g += sum(abs(mat_x/(EEs[ii,jj,-1] - EEs[ii,jj,:-1]))**2 + abs(mat_y/(EEs[ii,jj,-1] - EEs[ii,jj,:-1]))**2)*k_area
        T = trace_g/(2*pi)
        Evecs = Evecs[:,:,:,-1] 
        Ux = np.sum(Evecs.conjugate() * np.roll(Evecs,-1,axis=1), axis=2)
        Uy = np.sum(Evecs.conjugate() * np.roll(Evecs,-1,axis=0), axis=2)
    
    
        Ux = Ux/abs(Ux)
        Uy = Uy/abs(Uy)
        F12 = np.log(Ux * np.roll(Uy,-1,axis=1) / (Uy * np.roll(Ux,-1,axis=0)))[:-1,:-1]
        F12 = np.imag(F12)
        cn = np.sum(F12/(2*np.pi),axis=(0,1))
        print(Vd, T, cn, T-abs(cn))
        #sigma = np.sqrt(np.sum((F12/(2*np.pi)-1)**2, axis=(0,1)))
        #print(sigma)
        return T-abs(cn)
    
    
    dimH = 2* (2*Nk+1)**2
    T = quantum_metric_tensor(construct_ham,Q1,Q2,dimH,Vd)
    return T

def parallel_diag(Hlist, num_lowest = 0):
    solver = lambda H: eigh_sorted(H, num_lowest)
    return Parallel(n_jobs=num_cores)(delayed(solver)(H) for H in Hlist)

def EV_special(matrix):
    E, V = eigsh(matrix,k=1,which='SA',sigma=None)
    return V.flatten()

    

def plot_bands(EElist,klist,kcoord,klabels,energy_range=[-200,100]):
    numofk = len(klist)
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(np.arange(numofk),EElist,'k-',linewidth=1)
    ax.set_ylabel(r'$E/\mathrm{eV}$')
    ax.set_ylim(energy_range)
    ax.set_xlim([0,len(klist)])
    ax.set_xticks(kcoord)
    ax.set_xticklabels(klabels)
    return fig,ax

def g_vecs(theta, a0 = 3.52):
    g1 = np.array([8*pi*np.sin(theta/2)/(sqrt(3)*a0),0])
    gvecs = []
    for j in range(6):
        gvecs.append(rot_mat(j*pi/3)@g1)
    return np.asarray(gvecs)

def reciprocal_vecs(gvecs):
    return gvecs[0],gvecs[2]

def rot_mat(theta):
    return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

def klines(A,B,numpts):
    return np.array([np.linspace(A[jj],B[jj],numpts) for jj in range(len(A))]).T

def get_matrixEle(configs, k1234, V1234):
    config_lookup = {value: index for index, value in enumerate(configs)}
    Matele = []
    row = []
    col = []
    for ks,V in zip(k1234,V1234):
        k1, k2, k3, k4 = ks[0], ks[1], ks[2], ks[3]
        config_k3 =  getspin(configs,k3)
        valid_config_k3id = (config_k3==1).nonzero()[0]
        f3, new_config = c_an(configs[valid_config_k3id], k3)
        valid_config_k4id = (getspin(new_config,k4)==1).nonzero()[0]
        f4, new_config = c_an(new_config[valid_config_k4id], k4)
        f34 = f3[valid_config_k4id]*f4
        valid_config_k2id = (getspin(new_config,k2)==0).nonzero()[0]
        f2, new_config = c_dag(new_config[valid_config_k2id], k2)
        f2 = f34[valid_config_k2id]*f2
        valid_config_k1id = (getspin(new_config,k1)==0).nonzero()[0]
        f1, new_config = c_dag(new_config[valid_config_k1id], k1)
        fsign = f2[valid_config_k1id]*f1
        oldconfig_id = valid_config_k3id[valid_config_k4id[valid_config_k2id[valid_config_k1id]]]
        newconfig_id = np.array([config_lookup[nc] for nc in new_config]).astype(int)
        col.extend(oldconfig_id)
        row.extend(newconfig_id)
        Matele.extend(fsign * V * 0.5)
    return row, col, Matele

@njit
def convert_to_1_id(k1k2list, N):
# with C ordering
    return k1k2list[:,0]*N + k1k2list[:,1]

@njit
def convert_to_2_id(klist, N):
# with C ordering
    x,y = np.divmod(klist,N)
    return np.vstack((x,y)).T
# def construct_mat_element(basis_sector):

@njit
def two_index_unique(testarray,N):
    array_1d = convert_to_1_id(testarray,N)
    unique_id = np.unique(array_1d)
    unique_indices = List()
    for uid in unique_id:
        unique_indices.append((array_1d==uid).nonzero()[0])
    unique_2d = convert_to_2_id(unique_id, N)
    return unique_2d, unique_indices

def eigh_sorted(A, num_lowest=0):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues[-num_lowest:], eigenVectors[:,-num_lowest:]
              
def getspin(b,i):   #spin (0 or 1) at 'i' in the basis 'b'
    return (b>>i) & 1

def bitflip(b,i):  #Flips bits (1-->0, 0-->1) at loc 'i'
    return b^(1<<i)

def lcounter(b,i):  #left_counter: # of '1' in b left to site i
    num = b>>(i+1)
    #return bin(num).count('1')
    return np.array([bin(n).count('1') for n in num])

def c_an(configs, site):
    newconfigs = bitflip(configs, site)
    fsign =  1-2*(lcounter(configs, site)%2)
    return fsign, newconfigs

def c_dag(configs, site):
    newconfigs = bitflip(configs, site)
    fsign =  1-2*(lcounter(configs, site)%2)
    return fsign, newconfigs

def spinless_fsigns(config,indices): #for spinless config Ck1dag Ck2dag Ck4 Ck3
    i = min(indices[0],indices[1])
    j = max(indices[0],indices[1])
    k = min([indices[2],indices[3]])
    l = max([indices[2],indices[3]])
    Fkl = np.array([getspin(config,x) for x in range(k+1,l)])
    Fkl = np.prod(1-2*Fkl)
    new_config = bitflip(bitflip(config,k),l)
    Fij = np.array([getspin(new_config,y) for y in range(i+1,j)])
    Fij = np.prod(1-2*Fij)
    new_config = bitflip(bitflip(new_config,i),j)
    fsign = 1
    if indices[3] < indices[2]:
        fsign *= -Fkl
    else:
        fsign *= Fkl

    if indices[0] < indices[1]:
        fsign *= Fij
    else:
        fsign *= -Fij

    return fsign, new_config

def bitstring_config(sys_size, num_particle):
    decimal_numbers = []
    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ['0'] * sys_size
        for idx in ones_indices:
            binary_str[idx] = '1'
        decimal_numbers.append(int(''.join(binary_str), 2))
    return decimal_numbers

def flatten_list(nested_list):
    return np.array([element for sublist in nested_list for element in sublist])

def plot_charge_density(nk, ktot, Nup):
    num_K = ktot.shape[0]
    fig1, ax1 = plt.subplots()
    ax1.scatter(np.arange(num_K),nk, color="blue", marker="o",s=20, alpha=.8)
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r"$k_1+N_1 k_2$", fontsize = 12)
    ax1.set_ylabel(r"$\langle n_k \rangle$", fontsize = 12)
    ax1.axhline(y=Nup/num_K, color='r', linestyle='--')
    return fig1, ax1

import csv 

#theta = float(sys.argv[1])
Vd = 0#float(sys.argv[2])*0.2
#result= theta,round(Vd,2),T
thetalist = np.linspace(1,4,20)
for theta in thetalist:
    T = main(theta,Vd)
    result= theta,T
    #with open('trace_condition_theta='+str(round(theta,2))+'.csv', 'a') as f:
    with open('trace_condition.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result)

