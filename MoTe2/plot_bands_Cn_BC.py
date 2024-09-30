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

def main(theta, Vd):
    theta = theta/180*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2

    Nk = 4 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3

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
    

    def bandstructure(chern):
        npts = 40
        klist,kcoord,klabels = special_kline(npts)
        ham_morie_list = []
        for k in klist:
            ham_morie_list.append(construct_ham(k))
        EElist = find_eigenvalues(ham_morie_list)
        EElist = np.array(EElist)
        #print('bandwidth of second lowest band = ',max(EElist[:,1]) - min(EElist[:,1]), 'meV')
        #print('bandgap relative to lowest and third lowest =', min(EElist[:,1])-max(EElist[:,0]),  min(EElist[:,2])-max(EElist[:,1]))
        fig,ax = plot_bands(EElist,klist,kcoord,klabels,chern)
        return fig,ax
    
 
    def electron_density(nbz):
        klist = bz_sampling(nbz,Q1,Q2)
        nx = 100
        ny = 100
        xlist = np.linspace(-200,200,nx)*1e-10 #meter
        ylist = np.linspace(-200,200,ny)*1e-10
        xx,yy = np.meshgrid(xlist,ylist)
        x = xx.flatten()
        y = yy.flatten()
        solver = lambda k: EV_special(construct_ham(k))
        EvecList = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in klist)
        expfac = 1j*(Qxlist[:,np.newaxis]*x[np.newaxis,:] + Qylist[:,np.newaxis]*y[np.newaxis,:])
        density_top = np.zeros((nx,ny))
        density_bot = np.zeros((nx,ny))
        for Evec in EvecList:
            Evec = Evec.flatten()
            wf_top = np.dot(Evec[:halfdim],exp(expfac))
            wf_bot = np.dot(Evec[halfdim:],exp(expfac))
            wf_top = np.reshape(wf_top,(nx,ny))
            wf_bot = np.reshape(wf_bot,(nx,ny))
            density_top += np.abs(wf_top)**2
            density_bot += np.abs(wf_bot)**2
        density = (density_top + density_bot)/(nbz**2)
        print('sum ntop, sum nbot',np.sum(density_top,axis=(0,1))/(nx*ny*nbz**2), np.sum(density_bot,axis=(0,1))/(nx*ny*nbz**2))
        #plot density
        cmap = 'hot'#'RdBu'
        fig,ax = plt.subplots(figsize=(3,2.45))
        plt.subplots_adjust(left=0.1,
               bottom=0.1,
               right=0.9,
               top=0.9,
               wspace=0.4,
               hspace=0.4)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x/a_M$',fontsize=12)
        ax.set_ylabel(r'$y/a_M$',fontsize=12)
        #ax.set_title('Top Layer')
        c = ax.pcolormesh(xx/aM,yy/aM,density,shading = 'gouraud',cmap = cmap)
        cbar = plt.colorbar(c, fraction=0.05, pad=0.02) #, label='Density '+r'$n(r)$')
        cbar.set_label(label='Density '+r'$n(r)$')
        return fig,ax
    
    # Chern number calculation
    dimH = 2* (2*Nk+1)**2 # Hamitonian dimension
    bandidx = [-1, -2, -3] # compute the lowest three bands' chern number
    cn, figb, axb, figc, axc = chernnum(30,dimH,construct_ham, Q1, Q2, bandidx)
    print("Chern numbers from bottom:", cn)
    # band structure
    fig, ax = bandstructure(cn)
    
    ## Electron/hole density at the top band (band_id = 0)
    #fig2, ax2 = electron_density(nbz = 20)

    def diagonalize(k):
        Ho = construct_ham(k)
        Eig = eigsh(Ho,k=3,which='SA',sigma=None,return_eigenvectors=False)
        return np.sort(Eig.real)

    def threeD_bandPlot(npt):
        klist = np.array([np.array([i,j]) for i in range(npt) for j in range(npt)])
        kval = (klist / np.array([npt,npt])) @ np.array([Q1, Q2])
        solver = lambda k: diagonalize(k)
        Eigs = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kval)
        Eigs = np.array(Eigs)
        
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = klist[:,0].reshape(npt,npt)
        Y = klist[:,1].reshape(npt,npt)
        Eo = Eigs[:,0].reshape(npt,npt)
        E1 = Eigs[:,1].reshape(npt,npt)
        E2 = Eigs[:,2].reshape(npt,npt)
        ax.plot_wireframe(X, Y, Eo, color='blue', alpha=0.6)
        ax.plot_wireframe(X, Y, E1, color='green', alpha=0.6)
        ax.plot_wireframe(X, Y, E2)
        ax.view_init(elev=3, azim=-35)
        ax.set_zlabel(r"$\epsilon_k (meV)$", fontsize=12)
        ax.set_xlabel(r"$k_x (arb.)$", fontsize=12)
        ax.set_ylabel(r"$k_y (arb.)$", fontsize=12)
        ax.text(20,30, Eo.min()+0.5, 'c='+str(round(cn[0])), fontsize=15, weight="bold")
        ax.text(20,30, E1.min()+1.5, 'c='+str(round(cn[1])), fontsize=15, weight="bold")
        ax.text(20,30, E2.min()+2.5, 'c='+str(round(cn[2])), fontsize=15, weight="bold")
        return fig, ax


    #fig, ax = threeD_bandPlot(30)
    return fig, ax
    #return figb, axb, figc, axc 

def EV_special(matrix):
    E, V = eigsh(matrix,k=1,which='SA',sigma=None)
    return V

def bz_sampling_k12(N):
    xlist = np.linspace(-1/2,1/2,N+1)
    xlist = xlist[:-1]
    k12list = np.array([np.array([x1,x2]) for x1 in xlist for x2 in xlist]) # first x2
    return k12list

def bz_sampling(N, Q1, Q2):
    k12list = bz_sampling_k12(N)
    kxylist = k12list @ np.array([Q1,Q2])
    return kxylist

def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

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
    

def plot_bands(EElist,klist,kcoord,klabels,chern):
    numofk = len(klist)
    #EElist = np.array(EElist)
    Erange_low = min(EElist[:,0]) - 1
    Erange_high = max(EElist[:,2])+2
    energy_range = [Erange_low,Erange_high]
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(np.arange(numofk),EElist,'k-',linewidth=1)
    ax.set_ylabel(r'$E/\mathrm{eV}$')
    ax.set_ylim(energy_range)
    ax.set_xlim([0,len(klist)])
    ax.set_xticks(kcoord)
    ax.set_xticklabels(klabels)
    ax.text(numofk/2,EElist[:,0][int(numofk/2)]+0.5,r'$C= $'+str(round(chern[0])))
    ax.text(numofk/2,EElist[:,1][int(numofk/2)]+0.5,r'$C= $'+str(round(chern[1])))
    ax.text(numofk/2,EElist[:,2][int(numofk/2)]+0.5,r'$C= $'+str(round(chern[2])))
    print('Ediff', min(EElist[:,1])- max(EElist[:,0]), min(EElist[:,2])-max(EElist[:,1]))
    print('bandwidth of 2nd flat band = ', max(EElist[:,1])-min(EElist[:,1]))

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

def threeD_bandPlot(npt, klist, Eigs):
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure(figsize=(4.,4.8))
    ax = fig.add_subplot(projection='3d')
    X = klist[:,0].reshape(npt,npt)
    Y = klist[:,1].reshape(npt,npt)
    Eo = Eigs[:,0].reshape(npt,npt)
    E1 = Eigs[:,1].reshape(npt,npt)
    E2 = Eigs[:,2].reshape(npt,npt)
    ax.plot_wireframe(X, Y, Eo, color='blue', alpha=0.6)
    ax.plot_wireframe(X, Y, E1, color='green', alpha=0.6)
    ax.plot_wireframe(X, Y, E2)
    ax.view_init(elev=5, azim=-45)
    ax.set_zlabel(r"$\epsilon_k (meV)$", fontsize=13)
    ax.set_xlabel(r"$k_x/|\vec{g_1}|$", fontsize=13)
    ax.set_ylabel(r"$k_y/|\vec{g_3}|$", fontsize=13)
    #ax.text(20,30, Eo.min()+0.5, 'c='+str(round(cn[0])), fontsize=15, weight="bold")
    #ax.text(20,30, E1.min()+1.5, 'c='+str(round(cn[1])), fontsize=15, weight="bold")
    #ax.text(20,30, E2.min()+2.5, 'c='+str(round(cn[2])), fontsize=15, weight="bold")
    plt.xticks([-0.5,0,0.5], fontsize=13)
    plt.yticks([-0.5,0,0.5], fontsize=13)
    ax.set_zticklabels(ax.get_zticks(), fontsize=13)
    #ax.set_zticklabels([-27.5,-22.5,-17.5,-12.5,-7.5], fontsize=13)
    print('bandgap=', E1.min()-Eo.max())
    print('bandwidth=', Eo.max()-Eo.min())
    return fig, ax

def plot_BerryCurvature(F12, NB):
    dq = 1./NB
    qq = np.arange(0,1,dq)
    klist = np.array([[qq[ii],qq[jj]] for jj in range(NB) for ii in range(NB)])
    Q1, Q2 = np.array([1,0]), np.array([-0.5, sqrt(3)/2])
    klist = klist@np.array([Q1,Q2])
    k1x = klist[:,0].reshape(NB,NB)
    k1y = klist[:,1].reshape(NB,NB)
    klists = klist-Q1-Q2 + dq 
    k2x = klists[:,0].reshape(NB,NB)
    k2y = klists[:,1].reshape(NB,NB)
    klists = klist-Q2 
    k3x = (klists[:,0]-dq/2).reshape(NB,NB)
    k3y = (klists[:,1]+dq).reshape(NB,NB)
    klists = klist-Q1
    k4x = (klists[:,0]+dq).reshape(NB,NB)
    k4y = (klists[:,1]).reshape(NB,NB)
    klists = klist+Q1
    k5x = (klists[:,0]-dq).reshape(NB,NB)
    k5y = (klists[:,1]).reshape(NB,NB)
    klists = klist-2*Q1-Q2
    k6x = (klists[:,0]+2*dq).reshape(NB,NB)
    k6y = (klists[:,1]+dq).reshape(NB,NB)
    cmap = 'Blues'
    fig,ax = plt.subplots(figsize=(2.2,2.2))
    plt.subplots_adjust(left=0.1,
               bottom=0.1,
               right=0.9,
               top=0.9,
               wspace=0.4,
               hspace=0.4)
    ax.set_aspect('equal')
    #ax.set_xlabel(r'$kx$')
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim([-0.66,0.66])
    ax.set_ylim([-0.68,0.68])
    #print("max", F12[:,:,0].max())
    ticks=[0,0.15,0.3]
    F12 = F12[:,:,0]*2*pi
    print("max", F12.max())
    c = ax.pcolormesh(k1x,k1y,F12,shading='gouraud',cmap = cmap,vmin=0)#,vmax=Max)
    c = ax.pcolormesh(k4x,k4y,F12,shading='gouraud',cmap = cmap,vmin=0)#,vmax=Max)
    c = ax.pcolormesh(k2x,k2y,F12,shading='gouraud',cmap = cmap,vmin=0)#,vmax=Max)
    c = ax.pcolormesh(k3x,k3y,F12,shading='gouraud',cmap = cmap, vmin=0)#,vmax=Max)
    c = ax.pcolormesh(k5x,k5y,F12,shading='gouraud',cmap = cmap, vmin=0)#,vmax=Max)
    c = ax.pcolormesh(k6x,k6y,F12,shading='gouraud',cmap = cmap, vmin=0)#,vmax=Max)
    cbar=fig.colorbar(c, fraction=0.05, pad=0.02,orientation='horizontal', ticks=ticks, location='top')
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_title(r'$2\pi |F_{12}|$', fontsize=14)

    k1 = np.array((2*Q1+Q2)/3)
    k_corner = []
    for i in range(6):
        kpoint = rot_mat(-i*60*pi/180.)@k1
        k_corner.append(kpoint)
    k_corner.append(k1)
    k_corner = np.array(k_corner)
    kk = k_corner
    for i in range(6):
        ax.plot([kk[i][0],kk[i+1][0]],[kk[i][1],kk[i+1][1]], color='k', linewidth=0.7,linestyle='--')
    
    #ax.set_xlabel(r'$k_x/\vec{g_1}$')
    ax.axis('off')
    return fig, ax

def chernnum(NB,dim,HH, Q1, Q2, bandidx=None):
    dq = 1./NB
    #qq = np.arange(-0.5,0.5+dq,dq)
    qq = np.arange(0.0,1.0+dq,dq) #good for berry curvature plot

    # Diagonalizing Hamiltonian at each k
    k12list = np.array([[qq[ii],qq[jj]] for ii in range(NB+1) for jj in range(NB+1)])
    kxylist = k12list @ np.array([Q1,Q2])
    solver = lambda k: eigh_sorted(HH(k))
    result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
    N12 = (NB+1)**2
    VV = np.array([result[jj][1] for jj in range(N12)]).reshape([NB+1,NB+1,dim,dim],order='C')
    ##EE = np.array([result[jj][0] for jj in range(N12)]).reshape([NB+1,NB+1,dim],order='C')
    EE = np.array([result[j][0] for j in range(N12)])[:,bandidx]
    
    figb, axb = threeD_bandPlot(NB+1, k12list, EE)
    
    if bandidx is not None:
        VV = VV[:,:,:,bandidx]

    U1 = np.einsum('abij,abij->abj', np.roll(VV,-1,axis=0).conj(), VV)
    #this is same as suming axis 2: np.sum(np.roll(VV,-1,axis=0).conj()*VV, axis=2)
    U2 = np.einsum('abij,abij->abj', np.roll(VV,-1,axis=1).conj(), VV)

    # Berry Curvature
    #we should not double count value at pi and -pi. Removing double count and computing berry curvature F12
    F12 = np.angle(U1[:-1,:-1,:]*U2[1:,:-1,:]*U1[:-1,1:,:].conjugate()*U2[:-1,:-1,:].conjugate())        ##- Note that all 
    #F12 = np.imag(np.log(U1[:-1,:-1,:]*U2[1:,:-1,:]*U1[:-1,1:,:].conjugate()*U2[:-1,:-1,:].conjugate()))## |these three expressions
    #F12 = np.imag(np.log(U1[:-1,:-1,:]*U2[1:,:-1,:]/(U1[:-1,1:,:]*U2[:-1,:-1,:])))                      ##- are equivalent
    cn = np.sum(F12,axis=(0,1))/(2*np.pi)
    print('chernnum = ', cn)
    figc, axc = plot_BerryCurvature(abs(F12), NB)

    for i in range(len(bandidx)):
        axb.text(.3,0.5, min(EE[:,i])+1+i**2*0.5, 'c='+str(round(-cn[i])), fontsize=12, weight="bold")
    return cn, figb, axb, figc, axc

'''
def calculate_chernnum(vecs, N1, N2, dim, bandID):
    Vec = vecs[:,:,bandID].reshape(N1,N2,dim)
    Ux = np.sum(Vec.conjugate() * np.roll(Vec,-1,axis=1),axis=2)
    Uy = np.sum(Vec.conjugate() * np.roll(Vec,-1,axis=0), axis=2)
    Ux = Ux/np.abs(Ux)
    Uy = Uy/np.abs(Uy)
    F12 = np.log(Ux * np.roll(Uy,-1,axis=1) / (Uy * np.roll(Ux,-1,axis=0)))##[:-1,:-1]
    F12 = np.imag(F12)/(2*np.pi)
    #print(F12)
    #print()
    #print(sum(F12[-1,:]))
    #print(sum(F12[:,-1]))
    cn = np.sum(F12,axis=(0,1))
    #print(cn-sum(F12[-1,:])-sum(F12[:,-1]))
    return cn
'''
'''
def chernnum(NB,dim,HH,Q1, Q2, bandidx=None):
    q1 = 1  # lattice constant along 1
    q2 = 1  # lattice constant along 2
    N1 = NB * q2  # Number of pts along 1
    N2 = NB * q1  # Number of pts along 2
    N12 = q1*q2*NB; # The length of the side of the sample (squre)
    #k1list = np.arange(N1)/N12#*2*np.pi/N12
    #k2list = np.arange(N2)/N12#*2*np.pi/N12
    k1list = np.linspace(-1,0,N1)
    k2list = np.linspace(-1,0,N1)

    # Diagonalizing Hamiltonian at each k
    k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(N2) for ii in range(N1)])
    kxylist = k12list @ np.array([Q1,Q2])
    solver = lambda k: eigh_sorted(HH(k))#[1] 
    result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
    Eigs = np.array([result[j][0] for j in range(N1*N2)])[:,bandidx]
    figb, axb = threeD_bandPlot(NB, k12list, Eigs)
    VV = np.array([result[j][1] for j in range(N1*N2)])
    solver = lambda bID: calculate_chernnum(VV, N1, N2, dim, bID)
    cn = Parallel(n_jobs=len(bandidx))(delayed(solver)(bID) for bID in bandidx)
    print(cn)
    
    VV = np.array([result[j][1] for j in range(N1*N2)]).reshape([N1,N2,dim,dim],order='F')
    #VV = np.array([result[j][1] for j in range(N1*N2)])
    #solver = lambda bID: calculate_chernnum(VV, N1, N2, dim, bID)
    #cn = np.array([Parallel(n_jobs=len(bandidx))(delayed(solver)(bID) for bID in bandidx)])
    #print(cn) 
    #sys.exit()

    #VV = np.array(Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)).reshape([N1,N2,dim,dim],order='F')
    if bandidx is not None:
        VV = VV[:,:,:,bandidx]
    
    # Computing U1 connection
    #U1 = np.squeeze(np.sum(VV.conj()*np.roll(VV,-1,axis=0),axis=2))
    U1 = np.sum(VV.conj()*np.roll(VV,-1,axis=0),axis=2)
    U2 = np.squeeze(np.sum(VV.conj()*np.roll(VV,-1,axis=1),axis=2))
    U1 = U1/np.abs(U1)
    U2 = U2/np.abs(U2)
    # Berry Curvature
    F12 = np.imag(np.log(U1*np.roll(U2,-1,axis=0)*np.conj(np.roll(U1,-1,axis=1))*np.conj(U2)))/(2*np.pi)
    cn = np.sum(F12,axis=(0,1))
    
    return cn, figb, axb #, fig, ax
'''

def eigh_sorted(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors
    
theta = float(sys.argv[1])
Vd = float(sys.argv[2])
#f3Db, a3Db, figc, axc = main(theta,Vd)
f2, ax2 = main(theta,Vd)

#f3Db.savefig("3DbandDispersion_theta="+str(round(theta,2))+".png",bbox_inches='tight',dpi=300)
#f3Db.savefig("3DbandDispersion_theta="+str(round(theta,2))+"_Vd="+str(Vd)+".pdf",bbox_inches='tight',dpi=300)
#figc.savefig("berry_curvature_theta="+str(round(theta,2))+"_Vd="+str(Vd)+".png",bbox_inches='tight',dpi=300)
#figc.savefig("berry_curvature_theta="+str(round(theta,2))+"_Vd="+str(Vd)+".pdf",bbox_inches='tight',dpi=300)
#f2.savefig("moire_chargedensity_theta="+str(round(theta,2))+"_Vd="+str(Vd)+".pdf",bbox_inches='tight',dpi=300)
plt.show()
