import numpy as np
from numpy import pi,sqrt,sin,cos,exp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from joblib import Parallel,delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

def main(kmesh,spin=1):
    #-------parameters---------------------
    theta = 4.4
    theta = theta*pi/180.0

    mass = 0.62#0.62  #in the unit of me
    me = 9.11e-31 # kg
    hbar = 1.0546e-34
    Joule_to_meV = 6.242e21  #meV
    prefactor = hbar**2/(2*mass*me)*Joule_to_meV*1e20 #note k in per Angstrom
    V = 11.2  #meV
    w = -13.3 #meV, tunneling amplitude
    phi = -91*pi/180.0 #phase in V
    ao = 3.52   #Angstrom
    aM = ao/theta
    #--------------------------------------- 

    g_vecs = Gvecs(aM)
    #reciprocal lattice vectors
    G1 = g_vecs[0]
    G2 = g_vecs[1]

    kappa_p = (G1+G2)/3
    kappa_m = (2*G1-G2)/3.0
    
    Ncutoff = 1
    g_list = make_g_list(Ncutoff, G1, G2)
    
    def tb_bilayer_Ham(k):
        dim = len(g_list)
        H_top = np.zeros((dim,dim),dtype=complex)
        H_tunnel = np.zeros((dim,dim),dtype=complex)
        for i in range(dim):
            g = g_list[i]
            for j in range(i,dim):
                gp = g_list[j]
                H_top[j,i] += phase_term(gp,g,g_vecs,phi)
                
                bterm, tterm = tunneling_term(gp,g,g_vecs)
                H_tunnel[j,i] += bterm
                H_tunnel[i,j] += tterm
        
        H_tunnel = w * H_tunnel
        H_top = -V * H_top 
        H_top += H_top.conj().T  

        H_bottom = H_top.conj() ##because phi-->-phi

        ksq_top = kinetic_term(g_list,k,kappa_p,spin)
        ksq_bottom = kinetic_term(g_list,k,kappa_m,spin)

        np.fill_diagonal(H_top,-prefactor*ksq_top)
        np.fill_diagonal(H_bottom,-prefactor*ksq_bottom)

        Hamiltonian = np.zeros((2*dim,2*dim),dtype=complex)
        Hamiltonian[:dim,:dim] = H_top
        Hamiltonian[dim:,dim:] = H_bottom
        if spin == -1:
            Hamiltonian[dim:,:dim] = H_tunnel 
            Hamiltonian[:dim,dim:] = H_tunnel.conj().T
        elif spin == 1:
            Hamiltonian[dim:,:dim] = H_tunnel.conj().T
            Hamiltonian[:dim,dim:] = H_tunnel
        else:
            raise ValueError("spin takes value +1 for up and -1 for down spin.")
        return Hamiltonian
            
    #top_vbs = 1 
    solver = lambda k: EVspecific(tb_bilayer_Ham(k))
    EigenSpectra =  Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kmesh)
    
    return EigenSpectra

def bz_sampling(N,g1,g2):
    xlist = np.linspace(-1.0/2,1.0/2,N+1)
    xlist = xlist[:-1]
    k12list = np.array([np.array([x1,x2]) for x1 in xlist for x2 in xlist])
    kxysample = k12list @ np.array([g1,g2])
    return kxysample

def make_g_list(Ncutoff,G1,G2):
    n1 = np.arange(-Ncutoff, Ncutoff+1)
    n2 = np.ones(len(n1),dtype = int)
    n1_list = np.kron(n1,n2)
    n2_list = np.kron(n2,n1)
    #computing g = n1*g1 + n2*g2 and putting them in array
    g_list = np.array([n*G1 for n in n1_list])\
            +np.array([m*G2 for m in n2_list])
    return g_list

def EVspecific(H,bidx=1):
        N = len(H[0])
        Eval, Evec = linalg.eigh(H,eigvals=(N-bidx,N-1))
        return Eval, Evec

def rot_operator(theta):
    return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

def Gvecs(aM):
    G1 = np.array([4*pi/(sqrt(3)*aM),0])
    gvecs = []
    for j in range(6):
        gvecs.append(rot_operator(j*pi/3) @ G1)
    return gvecs

def isin(vec,vecs):
    logic = False
    for vi in vecs:
        if np.allclose(vec,vi):
            logic = True
    return logic

def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def kinetic_term(glist,k,kspecial,spin):
    k_layer = glist + k - spin * kspecial
    k_layer = k_layer**2
    k_layer = np.array([sum(k_layer[l]) for l in range(k_layer.shape[0])])
    return k_layer

def phase_term(gp,g,g_vecs,phi):
    phase = 0
    for z, gv in enumerate(g_vecs):
        if np.allclose(gp-g,gv):
            phase += exp(1j*(-1)**z*phi)
    return phase

def tunneling_term(gp,g,g_vecs):
    g2 = g_vecs[1]
    g3 = g_vecs[2]
    ji_term = 0
    ij_term = 0
    if np.allclose(gp,g) or np.allclose(gp,g+g2) \
        or np.allclose(gp,g+g3):
            ji_term += 1
    elif np.allclose(gp,g-g2) or np.allclose(gp,g-g3):
            ij_term += 1
    return ji_term,ij_term

def kgrids(A,B,npts):
    return np.array([np.linspace(A[j],B[j],npts) for j in range(len(A))]).T

def plot_bands(EElist,klist,kcoord,klabels,energy_range=[-200,100]):
    fig,ax = plt.subplots(figsize=(4,3))
    ksep = np.array([sqrt(sum((klist[l+1]-klist[l])**2)) for l in \
        range(klist.shape[0]-1)])
    ksep = np.insert(ksep,0,0)
    kline = []
    for i in range(len(ksep)):
        kline.append(sum(ksep[:i+1]))

    ax.plot(kline,EElist,'k',linewidth=1)
    ax.set_ylabel(r'$E/\mathrm{eV}$', fontsize=14)
    ax.set_ylim(energy_range)
    ax.set_xlim([0,max(kline)])
    ax.set_xticks(kcoord)
    ax.set_xticklabels(klabels,fontsize=14)
    return fig,ax
