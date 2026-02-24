import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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

epsilon_r = 10.0 ##float(sys.argv[1])  #coulomb parameter
def main(theta,N1, N2, Vd):
    theta = theta/180.0*pi
    a0 = 3.52e-10 #meter
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    print('aM', aM)
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2

    Nk = 8 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3

    #n_k34 = N1*N2
    n_g = (2*Nk+1)**2
    #n_q =  n_g * n_k34
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
        V = 11.2 # meVfrom joblib import Parallel, delayed

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
    
    def velocities(kxylist):
        k12list = kxylist/np.array([N1,N2])
        kxylist = k12list @ np.array([Q1,Q2])
        solver = lambda k: eigh_sorted(construct_ham(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        Eigs = np.array([result[jj][0] for jj in range(len(kxylist))])[:,-1]
        Uvecs = np.array([result[jj][1] for jj in range(len(kxylist))])
        Uvecs = Uvecs[:,:,-1]
        
        Vx = []
        Vy = []
        for i, k in enumerate(kxylist):
            vx, vy = mat_ele_H_der(k, Uvecs[i], Uvecs[i])  ##in the unit of (1/hbar mev-m)
            vx, vy = vx.real*1e7, vy.real*1e7  # in (1/hbar ev-A)
            Vx.append(vx)
            Vy.append(vy)
        return np.array(Vx), np.array(Vy), Eigs


    Ns = N1*N2
    klist = sample_k_q(N1, N2)
    
    print("Ns", Ns)

    Vx, Vy, Eigs = velocities(klist)
    v2 = Vx**2 + Vy**2
    print("QFI bound for k=1 : ", np.sum(v2)/Ns, 2.0/3*np.sum(v2)/Ns)
    KK, fig, ax = hexagonalBZmap(N1, N2, klist, v2)
    plt.savefig("velocity_distribution.pdf", bbox_inches="tight")
    plt.show()
    
    '''
    k_prod = [1]
    densities = np.linspace(0.0,1,60)
    qfi_data = {k: [] for k in k_prod}
    
    indices = np.argsort(v2)[::-1]  #descending order
    v2 = v2[indices]
    k12 = k12list[indices]
    kx, ky = k12[:,0], k12[:,1]
    Np = Ns//3
    #plt.scatter(kx[:Np],ky[:Np], c=v2[:Np], cmap='Blues', s=20)
    
    for k in k_prod:
        for n in densities:
            Np = int(round(n*Ns))
            if Np > Ns / 2: Np = Ns - Np
            vels = v2[:Np]
            qfi = np.sum(vels)/Ns
            print(n, qfi)
            qfi_data[k].append(qfi)
    
    plt.show()
    exit()
    '''
    return 

def boundary(k):
    ll1 = np.array([1,sqrt(3)])
    ll2 = np.array([-1,sqrt(3)])
    logic = False
    if (k[0]<=0.5) & (k@ll1<=1) & (k@ll2<=1):
        logic = True
    return logic

import matplotlib.colors as mcolors

white_sky = mcolors.LinearSegmentedColormap.from_list(
    "white_sky",
    #["#ffffff", "#4FA3F7"]   # white to sky blue
    ["#ffffff", "#1f78d1"]   # white to sky blue
)
def hexagonalBZmap(N1, N2, Ktot, values=None):
    g1 = np.array([1,0])
    g2 = np.array([-1/2, sqrt(3)/2])
    k1 = (2*g1+g2)/3
    k2 = (g1-g2)/3
    k3 = (-2*g2-g1)/3
    k4 = -k1
    k5 = -k2
    k6 = -k3
    kk = np.array([k1,k2,k3,k4,k5,k6,k1]) #+ g1 + g2
    fig, ax = plt.subplots(figsize=(3.6,2.5))
    for i in range(6):
        ax.plot([kk[i][0],kk[i+1][0]],[kk[i][1],kk[i+1][1]], color='k')

    newK = (Ktot/np.array([N1, N2]))@np.array([g1,g2]) 
    KK = []
    i = 0
    for k in newK:
        if boundary(k)==False:
            ktemp = k - g1
            if boundary(ktemp)==True:
                k = ktemp
            else:
                ktemp = k - g2
                if boundary(ktemp)==True:
                    k = ktemp
                else:
                    k = k - g1 - g2
        KK.append(k)
    KK = np.array(KK)
    norm = plt.Normalize(vmin=0, vmax=np.max(values))
    if values is not None:
        triang = tri.Triangulation(KK[:,0], KK[:,1])
        #tpc = ax.tricontourf(triang, values, levels=100, cmap=white_sky, norm = norm,shading='gouraud',  edgecolors='none')
        tpc = ax.tripcolor(triang,values,cmap=white_sky,norm=norm,shading='gouraud',rasterized=True)
    
    #fig.colorbar(tpc, ax=ax, label=r"$v^2$")

    #ax.scatter(KK[:,0], KK[:,1],color='blue')
    ax.set_aspect('equal')
    ax.axis('off')
    return KK, fig, ax


def bz_sampling_k12(N):
    xlist = np.linspace(-1/2,1/2,N+1)
    xlist = xlist[:-1]
    k12list = np.array([np.array([x1,x2]) for x1 in xlist for x2 in xlist]) # first x2
    return k12list

def bz_sampling(N, Q1, Q2):
    k12list = bz_sampling_k12(N)
    kxylist = k12list @ np.array([Q1,Q2])
    return kxylist

def sample_k_q(N1,N2):
    k1list = np.arange(N1+1,dtype=np.int32) #*2*np.pi/N12
    k2list = np.arange(N2+1,dtype=np.int32) #*2*np.pi/N12
    k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(N2+1) for ii in range(N1+1)], dtype=np.int32) # first k1 then k2
    return k12list
    

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
    

theta = 2.0 ##float(sys.argv[2])
Nx, Ny = 32,32#60, 60
Vd = 0.0 ##float(sys.argv[3]) #displacement field
main(theta,Nx, Ny, Vd)
