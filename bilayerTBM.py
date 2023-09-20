import numpy as np
from numpy import pi,sqrt,sin,cos,exp
from sys_config import moire_length
import matplotlib.pyplot as plt
from scipy import linalg
from joblib import Parallel,delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

Ncutoff = 1
#-------parameters---------------------
#theta = 4.4
#theta = theta*pi/180.0
#ao = 3.52   #Angstrom
mass = 0.62#0.62  #in the unit of me

V = 11.2  #meV
w = -13.3 #meV, tunneling amplitude
phi = -91*pi/180.0 #phase in V

me = 9.11e-31 # kg
hbar = 1.0546e-34
Joule_to_meV = 6.242e21  #meV
prefactor = hbar**2/(2*mass*me)*Joule_to_meV*1e20 #note k in per Angstrom
#aM = ao/theta
aM = moire_length()
#--------------------------------------- 
def make_g_list(): # in the unit of Q1 and Q2
    n1 = np.arange(-Ncutoff, Ncutoff+1)
    n2 = np.ones(len(n1),dtype = int)
    n1_list = np.kron(n1,n2)
    n2_list = np.kron(n2,n1)
    return np.column_stack((n1_list, n2_list))

def rot_operator(alpha):
    return np.array([[cos(alpha),-sin(alpha)],[sin(alpha),cos(alpha)]])

def Gvecs():
    G1 = np.array([4*pi/(sqrt(3)*aM),0])
    gvecs = []
    for j in range(6):
        gvecs.append(rot_operator(j*pi/3) @ G1)
    return gvecs

def EVspecific(H,bidx=1):
        N = len(H[0])
        Eval, Evec = linalg.eigh(H,eigvals=(N-bidx,N-1))
        #Eval, Evec = np.linalg.eigh(H)
        return Eval, Evec

def isin(vec,vecs):
    logic = False
    for vi in vecs:
        if np.allclose(vec,vi):
            logic = True
    return logic

def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def kinetic_term(glist,k,kspecial,Q1,Q2):
    k_layer = glist @ np.array([Q1,Q2]) + k - kspecial
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

g_vecs = Gvecs()
#reciprocal lattice vectors
G1 = g_vecs[0] #actual G1, G2 vectors
G2 = g_vecs[1]

kappa_p = (G1+G2)/3
kappa_m = (2*G1-G2)/3.0

g_list = make_g_list() #in the unit of G1,G2
halfdim = len(g_list)

def V_n_T_terms():
    H_top = np.zeros((halfdim,halfdim),dtype=complex)
    H_tunnel = np.zeros((halfdim,halfdim),dtype=complex)
    for i in range(halfdim):
        g = g_list[i] @ np.array([G1,G2])
         
        for j in range(i,halfdim):
            gp = g_list[j] @ np.array([G1,G2])
            H_top[j,i] += phase_term(gp,g,g_vecs,phi)

            bterm, tterm = tunneling_term(gp,g,g_vecs)
            H_tunnel[j,i] += bterm
            H_tunnel[i,j] += tterm
    
    H_tunnel = w * H_tunnel
    H_top = -V * H_top
    #diag_ele = np.diagonal(H_top)
    #print(diag_ele)
    H_top += H_top.conj().T #- np.diag(diag_ele)
    H_bottom = H_top.conj() ##because phi-->-phi

    return H_top, H_bottom, H_tunnel


def tb_bilayer_Ham(k,H_top,H_bottom,H_tunnel):
    ksq_top = np.real(kinetic_term(g_list,k,kappa_p,G1,G2))
    ksq_bottom = np.real(kinetic_term(g_list,k,kappa_m,G1,G2))
    H_top = H_top.copy()  # Make a copy of H_top that is writable
    H_bottom = H_bottom.copy()    
    np.fill_diagonal(H_top,-prefactor*ksq_top)
    np.fill_diagonal(H_bottom,-prefactor*ksq_bottom)
    
    Hamiltonian = np.zeros((2*halfdim,2*halfdim),dtype=complex)
    Hamiltonian[:halfdim,:halfdim] = H_top
    Hamiltonian[halfdim:,halfdim:] = H_bottom
    Hamiltonian[halfdim:,:halfdim] = H_tunnel.conj().T
    Hamiltonian[:halfdim,halfdim:] = H_tunnel
    
    return Hamiltonian


def GS_Espectrum(kmesh,htop,hbottom,htunnel):
    solver = lambda k: EVspecific(tb_bilayer_Ham(k,htop,hbottom,htunnel))
    EigenSpectra =  Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kmesh)
    return EigenSpectra

if __name__ == "__main__":

    def kgrids(A,B,npts):
        return np.array([np.linspace(A[j],B[j],npts) for j in range(len(A))]).T

    def special_kpath(npts):
        gamma = np.array([0,0])
        m = G1/2.
        kline1 = kgrids(gamma,m,npts)
        kline2 = kgrids(m,kappa_p,npts)
        kline3 = kgrids(kappa_p,gamma,npts)
        klabels = [r"$\gamma$", r'$m$', r'$\kappa_+$', r'$\gamma$']
        k_dist = np.array([0,sqrt(sum((m-gamma)**2)),sqrt(sum((kappa_p-m)**2))\
            , sqrt(sum((kappa_p-gamma)**2))])
        kcoords = []
        for i in range(len(k_dist)):
            kcoords.append(sum(k_dist[:i+1]))
        klines = np.concatenate((kline1[:-1,:],kline2[:-1,:],kline3[:,:]))
        return klines, klabels, kcoords

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
    
    top, bot, tunn = V_n_T_terms()
    def bandStructure():
        npts = 40
        EnergyRange= [-150, 40]
        klist, klabels, kcoords = special_kpath(npts)
        Espectra = GS_Espectrum(klist,top,bot,tunn)
        EElist = [Espectra[i][0] for i in range(len(klist))]
        fig, ax = plot_bands(EElist, klist, kcoords, klabels, EnergyRange)
        ax.axvline(x=kcoords[1],linestyle="--",linewidth=0.8,c='purple')
        ax.axvline(x=kcoords[2],linestyle="--",linewidth=0.8,c='purple')
        return fig, ax

    fig1, ax1 = bandStructure()
    plt.show()
