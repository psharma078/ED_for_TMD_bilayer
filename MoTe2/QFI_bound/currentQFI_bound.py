import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import sin,cos,exp, pi, sqrt
from joblib import Parallel, delayed
import multiprocessing
from itertools import combinations, permutations, repeat
from numba import jit,njit
from numba.typed import List
#from pylanczos import PyLanczos
import time
import sys
import math

num_cores = multiprocessing.cpu_count()

def main(theta, Vd):
    theta = theta/180.0*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    print('aM', aM)
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2

    Nk = 8 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3

    n_g = (2*Nk+1)**2
    Nklist = np.arange(-Nk,Nk+1)
    Nkoneslist = np.ones(2*Nk+1)
    mmlist = np.kron(Nklist,Nkoneslist)
    nnlist = np.kron(Nkoneslist,Nklist)

    Qxlist = nnlist*Q1[0] + mmlist*Q2[0]
    Qylist = nnlist*Q1[1] + mmlist*Q2[1]

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
    ### kline Gamma -- k+ -- M -- Gamma -- k-
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

    '''
    def special_kline(npts):
        Gammapt = np.array([0.,0.])
        mpt = Q1/2
        kline1 = klines(Gammapt,mpt,npts)
        kline2 = klines(mpt,kappa_p,npts)
        kline3 = klines(kappa_p,Gammapt,npts)
        klabels = [r"$\gamma$", r'$m$', r'$\kappa_+$', r'$\gamma$']
        kcoord = np.array([0,(npts-1),2*(npts-1),3*(npts-1)+1])
        return np.concatenate((kline1[:-1,:],kline2[:-1,:],kline3[:,:])), kcoord, klabels
    '''

    def gen_layer_hamiltonian(k):
        me = 9.10938356e-31 # kg
        m = 0.6*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV
        kxlist_top = Qxlist + k[0] - kappa_p[0]
        kylist_top = Qylist + k[1] - kappa_p[1]
        kxlist_bottom = Qxlist + k[0] - kappa_m[0]
        kylist_bottom = Qylist + k[1] - kappa_m[1]
        # First add potential terms
        psi = 107.7/180.0*pi
        V = 20.8 # meVfrom joblib import Parallel, delayed

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

    def gen_tunneling():
        # w = -18.0 # meV
        w = -23.8
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

    def H_der(k):
        me = 9.10938356e-31 # kg
        m = 0.6*me
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
        return ham_x, ham_y  ## in the unit meV-m

    def mat_ele_H_der(k, psi0, psi1):
        ham_x, ham_y = H_der(k)
        return psi0.conj()@ham_x@psi1, psi0.conj()@ham_y@psi1
    
    def compute_bloch_state(klist):
        k12list = np.zeros_like(klist,dtype=np.float64)
        k12list[:,0] = klist[:,0]/N1
        k12list[:,1] = klist[:,1]/N2
        kxylist = k12list @ np.array([Q1,Q2])
        #################################################################
        ## Solve for up spin
        solver = lambda k: eigh_sorted(construct_ham(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2  #including layer index

        singleVec = np.array([result[jj][1] for jj in range(n_k34)])
        singleVec = singleVec[:,:,-1]

        EEup = np.array([result[jj][0] for jj in range(n_k34)]).reshape([N1,N2,dim],order='F')[:,:,-1] #[:,:,-1] for only the ground state
        print("single EEup:", EEup)
    
        return singleVec
    
    @njit
    def find_id_in_klist(braket_k):
        return braket_k[:,0]+N1*braket_k[:,1]
    @njit
    def find_id_in_glist(g_k):
        g_k += Nk
        return g_k[:,0]+(2*Nk+1)*g_k[:,1] 

    def bandstructure(energy_range=None):
        npts = 40
        klist,kcoord,klabels = special_kline(npts)
        ham_morie_list = []
        for k in klist:
            ham_morie_list.append(construct_ham(k))
        EElist = find_eigenvalues(ham_morie_list)
        fig,ax = plot_bands(EElist,klist,kcoord,klabels,energy_range)
        return fig,ax
    
    def velocities(npts):
        k1list = np.linspace(-0.5,0.5,npts+1)
        k2list = np.linspace(-0.5,0.5,npts+1)

        # Diagonalizing Hamiltonian at each k
        k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(npts+1) for ii in range(npts+1)])
        kxylist = k12list @ np.array([Q1,Q2])
        solver = lambda k: eigh_sorted(construct_ham(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        Uvecs = np.array([result[jj][1] for jj in range(len(kxylist))])
        Uvecs = Uvecs[:,:,-1]
        
        Vx = []
        Vy = []
        for i, k in enumerate(kxylist):
            vx, vy = mat_ele_H_der(k, Uvecs[i], Uvecs[i])  ##in the unit of (1/hbar mev-m)
            vx, vy = vx.real*1e7, vy.real*1e7  # in (1/hbar ev-A)
            Vx.append(vx)
            Vy.append(vy)
        return np.array(Vx), np.array(Vy)


    npts = 35
    Ns = (npts+1)**2
    print("Ns", Ns)
    Vx, Vy = velocities(npts)
    k_prod = [1,2,3,4,5, 6, 7]
    densities = np.linspace(0.0,1,60)
    qfi_data = {k: [] for k in k_prod}
    
    for k in k_prod:
        for n in densities:
            Np = int(round(n*Ns))
            qfi_data[k].append(max_QFI(Vx, k, Ns, Np))

    plt.figure(figsize=(6,5))
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(k_prod)))
    for k,c in zip(k_prod,colors):
        plt.plot(densities, np.array(qfi_data[k])/Ns, label=f"k = {k}", color=c)
    plt.legend(fontsize=12)
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel("QFI bound " + r"[$eV.Å$ ($e/\hbar$)]", fontsize=14)
    plt.tick_params(axis='both',direction='in',top=True,bottom=True,left=True,right=True, labelsize=12)
    plt.xlim(0,1)
    #plt.ylim(0,0.015)
    plt.grid(True)
    plt.savefig("qfi_bound_theta5_wangPara.pdf", bbox_inches="tight")
    plt.show()
    
    #solver = lambda gid: manybody_Hamiltonian(gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))
    #Evals = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configsGroupID)
     
    return #Ktot, Evals

def max_QFI(v_k, m, N_s, N_p):
    """
    Python implementation of max_QFI

    Args:
        v_k: Weight vector of momentum modes (list or numpy array)
        m: k-producibility (entanglement depth)
        N_s: Total number of modes
        N_p: Total number of particles
    """
    v_k = np.sort(np.array(v_k))  # Sort v_k in ascending order

    # Particle-hole duality check: if N_p > N_s/2
    if N_p > N_s/ 2:
        N_p = N_s - N_p

    # Calculate q and r
    r = N_p % m            # Remainder particles
    q = (N_p - r) // m     # Full blocks (integer division)

    # Calculate number of blocks
    num_blocks = int(np.ceil(N_p / m))

    p_plus = []
    # 1. Partition p_+ blocks (low-energy modes)
    for p in range(1, num_blocks + 1):
        if p <= q:
            # Indices for full blocks (0-indexed adjustment)
            indices = np.arange((p - 1) * m, p * m)
        else:
            # Indices for the remainder block
            indices = np.arange(q * m, N_p)
        p_plus.append(indices)

    # 2. Partition p_- blocks (high-energy modes)
    # Using the substitution: l_minus = N_s - l_plus + 1
    # In 0-indexing, this becomes: l_minus = (N_s - 1) - l_plus
    p_minus = []
    for indices in p_plus:
        indices_minus = (N_s - 1) - indices
        p_minus.append(indices_minus)

    # 3. Calculate Max QFI (sum of squared differences)
    max_qfi = 0.0
    for i in range(len(p_plus)):
        # Get modes for current block
        p_plus_modes = p_plus[i]
        p_minus_modes = p_minus[i]

        # Sum weights and compute square of the difference
        sum_v_plus = np.sum(v_k[p_plus_modes])
        sum_v_minus = np.sum(v_k[p_minus_modes])

        max_qfi += (sum_v_plus - sum_v_minus)**2

    return max_qfi

def bz_sampling_k12(N):
    xlist = np.linspace(-1/2,1/2,N+1)
    xlist = xlist[:-1]
    k12list = np.array([np.array([x1,x2]) for x1 in xlist for x2 in xlist]) # first x2
    return k12list

def bz_sampling(N, Q1, Q2):
    k12list = bz_sampling_k12(N)
    kxylist = k12list @ np.array([Q1,Q2])
    return kxylist

def sample_k_q(N1,N2,Nmax):
    k1list = np.arange(N1,dtype=np.int32) #*2*np.pi/N12
    k2list = np.arange(N2,dtype=np.int32) #*2*np.pi/N12
    k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(N2) for ii in range(N1)], dtype=np.int32) # first k1 then k2
    glist = np.arange(-Nmax,Nmax+1,dtype=np.int32)
    glist = np.array([np.array([glist[ii],glist[jj]]) for jj in range(2*Nmax+1) for ii in range(2*Nmax+1)], dtype=np.int32)
    glist[:,0] *= N1
    glist[:,1] *= N2
    qlist = np.tile(k12list, [(2*Nmax+1)**2,1]) + np.repeat(glist,N1*N2, axis=0)
    return k12list, qlist
    

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
    g1 = np.array([4*pi*theta/(sqrt(3)*a0),0])
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

def find_eigenvalues(hamlist):
    EElist = []
    for ham in hamlist:
        EE = np.linalg.eigvalsh(ham)
        EElist.append(EE)
    return EElist

def find_eigenvalues_eigenstates(hamlist):
    EElist = []
    EVlist = []
    for ham in hamlist:
        EE,EV = np.linalg.eigh(ham)
        idx = np.argsort(EE)
        EElist.append(EE[idx])
        EVlist.append(np.asfortranarray(EV[:,idx]))
    return EElist, EVlist

def config_array(configs,Nsys):
    configs_bin = list(map(lambda x: format(x, '0'+str(Nsys)+'b'), configs))
    #print("bin", configs_bin[0])
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])  #convert binary num to array

              
def eigh_sorted(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors
    

theta = 5.0 ##float(sys.argv[1])
Vd = 0.0 ##float(sys.argv[2]) #displacement field
main(theta, Vd)
