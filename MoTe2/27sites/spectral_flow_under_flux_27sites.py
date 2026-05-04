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
import csv

num_cores = multiprocessing.cpu_count()

epsilon_r = 10.0##float(sys.argv[1])  #coulomb parameter

def main(theta, Nup, Vd):
    theta = theta/180*pi
    a0 = 3.52e-10
    aM = a0/(2*np.sin(theta/2))
    gvecs = g_vecs(theta, aM)
    Q1,Q2 = reciprocal_vecs(gvecs)
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2
    
    T1, T2, mnklist = specialFor27sites(aM)
    klist_lookup = {tuple(value): index for index, value in enumerate(mnklist)} 
    Nk = 8 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3
    N1 = 9
    N2 = 3

    n_k34 = N1*N2
    n_g = (2*Nk+1)**2
    n_q =  n_g * n_k34
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

    def gen_layer_hamiltonian(k, delk):
        me = 9.10938356e-31 # kg
        m = 0.64*me   #pressure P=0
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV
        kxlist_top = Qxlist + (k[0] + delk[0]) - kappa_p[0]
        kylist_top = Qylist + (k[1] + delk[1]) - kappa_p[1]
        kxlist_bottom = Qxlist + (k[0] + delk[0]) - kappa_m[0]
        kylist_bottom = Qylist + (k[1] + delk[1]) - kappa_m[1]
        # First add potential terms
        psi = -90.08/180.0*pi
        V = 8.621 # meV

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
        ## w = -18.0 # meV
        w = -8.35
        Delta_T = -w*(delta_nn_mm + delta_nnm1_mmm1 + delta_nn_mmm1)
        return Delta_T

    def construct_ham(k, delk):
        H_top,H_bottom = gen_layer_hamiltonian(k,delk)
        Delta_T = gen_tunneling()
        ham = np.zeros((ham_dim,ham_dim),dtype=np.complex128)
        ham[:halfdim,:halfdim] = H_top
        ham[halfdim:,halfdim:] = H_bottom
        ham[:halfdim,halfdim:] = Delta_T.conj().T
        ham[halfdim:,:halfdim] = Delta_T #.conj().T
        return ham
    
    def compute_F_mat(dk):
        #k12list = np.zeros_like(klist,dtype=np.float64)
        #k12list[:,0] = klist[:,0]/N1
        #k12list[:,1] = klist[:,1]/N2
        #kxylist = k12list @ np.array([Q1,Q2])
        kxylist = mnklist @ np.array([T1, T2])
        #################################################################
        ## Solve for up spin
        solver = lambda k: eigh_sorted(construct_ham(k,dk))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2  #including layer index
        
        singleVec = np.array([result[jj][1] for jj in range(n_k34)])
        singleVec = singleVec[:,:,-1]

        EEup = np.array([result[jj][0] for jj in range(n_k34)])##.reshape([N1,N2,dim],order='F')[:,:,-1] #[:,:,-1] for only the ground state
        #print("single k, EEup")
        #for x in range(len(mnklist)):
        #    print(mnklist[x], EEup[x][-1])
        EEup = EEup.reshape([N1,N2,dim],order='F')[:,:,-1]
    
        VV = np.array([result[jj][1] for jj in range(n_k34)], dtype=np.complex128).reshape([N1,N2,2*Nk+1, 2*Nk+1, 2, dim],order='F')[:,:,:,:,:,-1]
        Fmat = np.zeros([N1,N2,N1,N2,2*Nk+1,2*Nk+1], dtype=np.complex128)
        Fmat[:,:,:,:,Nk,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:,:].conj(), VV[:,:,:,:,:])
            ###########################################################
        for jj in range(1,Nk+1):
            for ii in range(1,Nk+1):
                Fmat[:,:,:,:,Nk+ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,jj:,:].conj(), VV[:,:,:(-ii),:(-jj),:])
                Fmat[:,:,:,:,Nk+ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,:(-jj):,:].conj(), VV[:,:,:(-ii),jj:,:])
                Fmat[:,:,:,:,Nk-ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),jj:,:].conj(), VV[:,:,ii:,:(-jj),:])
                Fmat[:,:,:,:,Nk-ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),:(-jj),:].conj(), VV[:,:,ii:,jj:,:])
        for jj in range(1,Nk+1):
            Fmat[:,:,:,:,Nk+jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,jj:,:,:].conj(), VV[:,:,:(-jj),:,:])
            Fmat[:,:,:,:,Nk-jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-jj),:,:].conj(), VV[:,:,jj:,:,:])
            Fmat[:,:,:,:,Nk,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,jj:,:].conj(), VV[:,:,:,:(-jj),:])
            Fmat[:,:,:,:,Nk,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:(-jj),:].conj(), VV[:,:,:,jj:,:])
        Fmat_up = Fmat.reshape(N1*N2,N1*N2,(2*Nk+1)**2,order='F')
        #################################################################
        return Fmat_up.flatten(order='F'), EEup, singleVec
    
    def find_id_in_klist(braket_k):
        kid = np.array([klist_lookup[tuple(bk)] for bk in braket_k]) 
        return kid

    
    def find_id_in_glist(g_k):
        g_k += Nk
        return g_k[:,0]+(2*Nk+1)*g_k[:,1]
    
    #################################
    ## CHECK INDICES
    fname1 = 'k3plus_q_nk=8_nh=9.txt'
    fname2 = 'k4minus_q_nk=8_nh=9.txt'
    k3_p_q, g_k3pq, k3q_braket = readfile(fname1)
    k4_m_q, g_k4mq, k4q_braket = readfile(fname2)
    k3q_lookup = {tuple(value): index for index, value in enumerate(k3_p_q)}
    k4q_lookup = {tuple(value): index for index, value in enumerate(k4_m_q)}
    
    def get_k_q_ids(k3_id,k4_id,mnklist,qlist):
        k3_q = mnklist[k3_id] + qlist
        k4_q = mnklist[k4_id] - qlist
        k3qID = np.array([k3q_lookup[tuple(k3q)] for k3q in k3_q])
        k4qID = np.array([k4q_lookup[tuple(k4q)] for k4q in k4_q])
        g_k3_q = g_k3pq[k3qID]
        braket_k3_q = k3q_braket[k3qID]
        g_k4_q = g_k4mq[k4qID]
        braket_k4_q = k4q_braket[k4qID] 
        k3_q_id = find_id_in_klist(braket_k3_q)
        k4_q_id = find_id_in_klist(braket_k4_q)
        g_k3_q_id = find_id_in_glist(g_k3_q)
        g_k4_q_id = find_id_in_glist(g_k4_q)
        return k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id
    
    def compute_coulomb(k3_id,k4_id,mnklist,qlist, Fmat1,Fmat2, Vlist):
        k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id = get_k_q_ids(k3_id,k4_id,mnklist,qlist)
        ind_k = (g_k3_q_id<n_g) & (g_k3_q_id>=0) & (g_k4_q_id<n_g) & (g_k4_q_id>=0) & (k3_q_id < k4_q_id)
        k3_q_id = k3_q_id[ind_k]
        k4_q_id = k4_q_id[ind_k]
        V = Fmat1[g_k3_q_id[ind_k]*n_k34**2 + k3_id*n_k34 + k3_q_id] * Fmat2[g_k4_q_id[ind_k]*n_k34**2 + k4_id*n_k34 + k4_q_id] * Vlist[ind_k] 
        k1k2list = np.vstack((k3_q_id,k4_q_id)).T
        k1k2, k1k2_ids =  two_index_unique(k1k2list,n_k34)
        Vk1k2 = List()
        for k_id in k1k2_ids:
            Vk1k2.append(np.sum(V[k_id]))
        #print(Vk1k2)
        return Vk1k2, k1k2

    def coulomb_potential(qlist):
        zeroidx = np.nonzero(np.prod(qlist==0,axis=1))[0][0]
        #qxy = np.zeros_like(qlist,dtype=np.float64)
        #qxy[:,0] = qlist[:,0]/N1
        #qxy[:,1] = qlist[:,1]/N2
        qxy = qlist @ np.array([T1,T2]) # in m^-1
        absqlist = np.linalg.norm(qxy,axis=1) 
        k0 = 8.99e9
        J_to_meV = 6.242e21  #meV
        e_charge = 1.602e-19 # coulomb
        #epsilon = 8.854e-12
        #epsilon_r = 5
        Area = np.sqrt(3)/2*N1*N2*aM**2
        Vc = 2*np.pi*e_charge**2/(epsilon_r*absqlist)*J_to_meV/Area*k0   
        Vc[zeroidx]=0
    
        return Vc
         

    def create_table_k1234_from_k3k4(ii,jj,mnklist, qlist, Fmat_1, Fmat_2, Vlist):
        k1k2k3k4 = []
        Vk1k2, k1k2 = compute_coulomb(ii,jj,mnklist,qlist,Fmat_1, Fmat_2, Vlist)
        for kk in k1k2:
            k1k2k3k4.append(np.concatenate((kk,np.array([ii,jj]))))
        return k1k2k3k4, Vk1k2
    
    def coulomb_matrix_elements(mnklist,qlist, Vlist, Fmat_up):
        k1234_upup = []
        Vc_upup = []
        for ii in range(n_k34):
            for jj in range(ii):
                ##########################
                ### ii neq jj
                _,Vk1234_temp = create_table_k1234_from_k3k4(ii,jj,mnklist,qlist,Fmat_up, Fmat_up, Vlist)
                #k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(jj,ii,mnklist,qlist, Fmat_up, Fmat_up, Vlist)
                k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                Vc_temp = []
                for ll in range(len(Vk1234)):
                    Vc_temp.append(2*(Vk1234[ll] - Vk1234_temp[ll]))
                Vc_upup.append(Vc_temp)
        return k1234_upup, Vc_upup
    
    def create_coulomb_table(k1234_upup, Vc_upup):
        return np.vstack(k1234_upup), np.hstack(Vc_upup)

    def diagonalization(klist, qlist, Vlist, binary_lists, configGID, dk):                  
        Fmat_up, EEup, singleVec = compute_F_mat(dk)
        singleE = binary_lists @ EEup.flatten(order='F')   
        k1234_upup, Vc_upup = coulomb_matrix_elements(mnklist,qlist, Vlist, Fmat_up)
        k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)
        Eigs = []
        Len = len(configGID)
        for gid in configGID:
            Eigs.append(manybody_Hamiltonian(Len, gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128)))
        if Len==1: Eigs = Eigs[0]
        return Eigs
        
    qlist, klist = sample_q(N1,N2,Nk,mnklist)
    Vlist = coulomb_potential(qlist)
    #Fmat_up, EEup = compute_F_mat()
    
    #k1234_upup, Vc_upup = coulomb_matrix_elements(mnklist,qlist, Vlist, Fmat_up)
    #k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)
    
    up_configs = np.array(bitstring_config(n_k34,Nup))
    binary_lists = np.fliplr(config_array(up_configs,N1*N2))
    Ktot, configsGroupID = groupingKsum(binary_lists, mnklist, N1, N2, Nup, Nk)
   
    #constructing dk grids
    dTheta = np.linspace(0, 3, 40) 
    dTheta_grid = np.array([np.array([th1, 0]) for th1 in dTheta]) 
    delta_klist = dTheta_grid @ np.array([T1, T2])

    Gs_Ktot_ind = [5]
    configGID = []
    configurs = []
    for g in Gs_Ktot_ind:
        configid = configsGroupID[g]
        configGID.append(configid)
        configurs.append(up_configs[configid])
    
    solver = lambda dk: diagonalization(klist, qlist, Vlist, binary_lists, configGID, dk)
    results = Parallel(n_jobs=41)(delayed(solver)(dk) for dk in delta_klist)
    #results = Parallel(n_jobs=len(dTheta_grid))(delayed(solver)(dk) for dk in delta_klist)
    for i in range(len(dTheta)):
        print(dTheta[i],results[i])

    Eigs = np.array(results)
    Eigs = Eigs - Eigs.min()
    print()
    print(dTheta)
    print(Eigs)

    return dTheta,Eigs

def readfile(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()

    # Process the data
    xval = []
    g_x = []
    x_braket = []
    for line in data:
        line = line.strip().strip('[]').split('] [')
        line = np.array([np.array((line[i].strip('[]').split())).astype(int) for i in range(3)])
        xval.append(line[0])
        g_x.append(line[1])
        x_braket.append(line[2])
    return np.array(xval), np.array(g_x), np.array(x_braket)

def specialFor27sites(aM):
    a1 = aM*np.array([sqrt(3)/2,0.5])
    a2 = aM*np.array([0,1])
    L1 = 3*(2*a1-a2)
    L2 = 3*(2*a2-a1)
    zhat = np.array([0,0,1])
    A = np.cross(L1,L2)
    T1 = (2*pi*np.cross(L2,zhat)/A)[:-1]
    T2 = (-2*pi*np.cross(L1,zhat)/A)[:-1]
    To = np.array([0, 0])
    uc = np.array([To, T1, T2])
    t1 = 2*T2-T1
    t2 = 2*T1-T2
    k1list = np.arange(3)
    k2list = np.arange(3)
    k12list = np.array([np.array([i,j]) for i in k1list for j in k2list])
    k12 = k12list@np.array([t1,t2])

    klist = np.array([uc+k for k in k12]).reshape(27,2)
    mrange = np.concatenate((np.arange(7),np.array([-1,-2,-3])))
    mnlist = []
    for m in mrange:
        for n in mrange:
            a = m*T1 + n*T2
            for k in klist:
                if np.allclose(a,k,atol=1e-6):
                    mn = np.array([m,n])
                    mnlist.append(mn)
    
    return T1, T2, np.array(mnlist)

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

def glist_gtrans(Nmax,Nc):
    p1 = np.array([6,-3])
    p2 = np.array([-3,6])
    glist = np.arange(-Nc*Nmax,Nc*Nmax+1,dtype=np.int32)
    glist = np.array([np.array([glist[ii],glist[jj]]) for jj in range(2*Nc*Nmax+1) for ii in range(2*Nc*Nmax+1)], dtype=np.int32)
    gtrans = glist@np.array([p1,p2])
    return glist, gtrans

def find_gindex(glist, gtrans, qf, mnklist):
    qi = qf - gtrans
    qi_m_klist = qi[:, None, :] - mnklist
    qnorm = np.linalg.norm(qi_m_klist, axis=2).astype(int)
    gindex = (qnorm==0).nonzero()[0]
    if len(gindex)!=1:
    	raise ValueError('Empty index value encountered.')
    	sys.exit()
    x_braket = qi[gindex].flatten()
    g_q = glist[gindex].flatten()
    #print(qf, g_q, x_braket)
    return g_q, x_braket

def compute_reduce_k(q_list, mnklist, glist, gtrans):
    solver = lambda qf: find_gindex(glist, gtrans, qf, mnklist)
    result = np.array(Parallel(n_jobs=num_cores)(delayed(solver)(qf) for qf in q_list))
    g_q = result[:,0]
    q_braket = result[:,1]
    return g_q, q_braket

def sample_q(N1,N2,Nmax, mnklist):
    k1list = np.arange(N1,dtype=np.int32) #*2*np.pi/N12
    k2list = np.arange(N2,dtype=np.int32) #*2*np.pi/N12
    k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(N2) for ii in range(N1)], dtype=np.int32) # first k1 then k2
    glist = np.arange(-Nmax,Nmax+1,dtype=np.int32)
    glist = np.array([np.array([glist[ii],glist[jj]]) for jj in range(2*Nmax+1) for ii in range(2*Nmax+1)], dtype=np.int32)
    p1 = np.array([6,-3])
    p2 = np.array([-3,6])
    glist1 = np.array([g*p1 for g in glist[:,0]])
    glist2 = np.array([g*p2 for g in glist[:,1]])
    glist12 = glist1 + glist2
    qlist = np.tile(mnklist, [(2*Nmax+1)**2,1]) + np.repeat(glist12,N1*N2, axis=0)
    return qlist, k12list
    

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

def g_vecs(theta, aM):
    g1 = np.array([4*pi/(sqrt(3)*aM),0])
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

def chernnum(NB,dim,HH,Q1, Q2, bandidx=None):
    q1 = 1  # lattice constant along 1
    q2 = 1  # lattice constant along 2
    N1 = NB * q2  # Number of pts along 1
    N2 = NB * q1  # Number of pts along 2
    N12 = q1*q2*NB; # The length of the side of the sample (squre)
    k1list = np.arange(N1)/N12#*2*np.pi/N12
    k2list = np.arange(N2)/N12#*2*np.pi/N12
  
    # Diagonalizing Hamiltonian at each k
    k12list = np.array([np.array([k1list[ii],k2list[jj]]) for jj in range(N2) for ii in range(N1)])
    kxylist = k12list @ np.array([Q1,Q2])
    solver = lambda k: eigh_sorted(HH(k))[1]
    VV = np.array(Parallel(n_jobs=6)(delayed(solver)(k) for k in kxylist)).reshape([N1,N2,dim,dim],order='F')
    if bandidx is not None:
        VV = VV[:,:,:,bandidx]

    # Computing U1 connection
    U1 = np.squeeze(np.sum(VV.conj()*np.roll(VV,-1,axis=0),axis=2))
    U2 = np.squeeze(np.sum(VV.conj()*np.roll(VV,-1,axis=1),axis=2))
    U1 = U1/np.abs(U1)
    U2 = U2/np.abs(U2)
    # Berry Curvature
    F12 = np.imag(np.log(U1*np.roll(U2,-1,axis=0)*np.conj(np.roll(U1,-1,axis=1))*np.conj(U2)))/(2*np.pi)
    cn = np.sum(F12,axis=(0,1))
    return cn


def config_array(configs,Nsys):
    configs_bin = list(map(lambda x: format(x, '0'+str(Nsys)+'b'), configs))
    #print("bin", configs_bin[0])
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])  #convert binary num to array

def groupingKsum(binary_lists, klist, N1, N2, Nup, Nk):
    #dimH = len(configs)
    #occ_ksum = binary_lists @ klist
    #glist, gtrans = glist_gtrans(Nk,2)
    #_,occ_ksum = compute_reduce_k(np.array(occ_ksum), klist, glist, gtrans)
    _,_,occ_ksum = readfile('reduced_Ksum27_nh='+str(Nup)+'.txt')
    occ_ksum += np.array([2,2])
    Ktot, grouped_k = two_index_unique(occ_ksum,N1*N2)
    Ktot -= np.array([2,2])
    return Ktot,  grouped_k


def newconfig_matel(config,k_index,Vcoul):
    fsign, newconfiguration = spinless_fsigns(config,k_index)
    mat_el = 0.5 * fsign * Vcoul
    return newconfiguration, mat_el

def get_matrixEle(configs, k1234, V1234):
    config_lookup = {value: index for index, value in enumerate(configs)}
    Matele = []
    row = []
    col = []
    for ks,V in zip(k1234,V1234):
        configs_k3 = getspin(configs, ks[2])
        configs_k4 = getspin(configs, ks[3])
        valid_k34_id = ((configs_k3*configs_k4)==1).nonzero()[0]

        configs_k1 = getspin(configs[valid_k34_id], ks[0])
        configs_k2 = getspin(configs[valid_k34_id], ks[1])
        valid_kid1 = ((configs_k1+configs_k2)==0).nonzero()[0]

        valid_kid2 = (((ks[0]==ks[2]) or (ks[0]==ks[3])) & (configs_k2==0)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k34_id[valid_kid1], valid_k34_id[valid_kid2]))

        valid_kid3 = (((ks[1]==ks[2]) or (ks[1]==ks[3])) & (configs_k1==0)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid3]))

        valid_kid4 = ((ks[0]==ks[2]) & (ks[1]==ks[3]) & (configs_k2==1)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid4]))

        valid_kid5 = ((ks[0]==ks[3]) & (ks[1]==ks[2]) & (configs_k2==1)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid5]))

        for jj in valid_k_id:
            config = configs[jj]
            fsign, newconfig = spinless_fsigns(config,ks)
            Matele.append(fsign * V * 0.5)
            col.append(jj)
            row.append(config_lookup[newconfig])
    return row, col, Matele

def manybody_Hamiltonian(Len, configs_indx, up_configs, singleE, k1234, V1234):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))
    
    # shape = (dimHam,dimHam)
    Eigs = []
    if Len==1:
        Eigs = eigsh(matrix,k=3,which='SA',sigma=None,return_eigenvectors=False)
        Eigs = np.sort(Eigs.real)
    elif Len>1:
        Eigs = min(eigsh(matrix,k=1,which='SA',sigma=None,return_eigenvectors=False))
    return Eigs

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

              
def eigh_sorted(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors
    
def plot_manybody_energies(klist,N1, Evals):
    k1 = klist[:,0]
    k2 = klist[:,1]
    xvals = k1 + N1 * k2
    Egs = np.min(np.concatenate(Evals))
    fig,ax =  plt.subplots(figsize=(4,3))
    for i, eig in enumerate(Evals):
        x = np.full(len(eig), xvals[i])
        ax.scatter(x, np.array(eig)-Egs, c='blue')
    ax.set_xlabel("$k_1 + N_1 k_2$")
    ax.set_ylabel("$E-E_{GS}$")
    ax.set_ylim([-0.1,5.1])
    #plt.savefig("Energy_N=5x3_nup=3.pdf",dpi=300,bbox_inches='tight')
    return fig, ax

def getspin(b,i):   #spin (0 or 1) at 'i' in the basis 'b'
    return (b>>i) & 1

def bitflip(b,i):  #Flips bits (1-->0, 0-->1) at loc 'i'
    return b^(1<<i)

def lcounter(b,i):  #left_counter: # of '1' in b left to site i
    num = b>>(i+1)
    return bin(num).count('1')


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

def plot_ED_energies(Ktot, Evals):
    Evals_array = np.array(Evals)
    print("ktot,         Eigen values (meV)")
    for i in range(len(Ktot)):
        print(Ktot[i], Evals[i])
    minE = Evals_array.min()
    maxE = Evals_array.max()
    print()
    print("Eigenvalues sorted")
    print(np.sort(Evals_array.flatten()))
    k1Dlist = np.array([20,26,10,15,23,0,8,13,18,25,3,11,16,21,1,6,14,19,24,4,9,17,22,2,7,12,5])
    fig, ax = plt.subplots(figsize=(3,2.5))
    ax.plot(k1Dlist, (Evals_array - minE),'k.',markersize=10)
    ax.set_ylabel(r'$E - E_{GS}(meV)$', fontsize=12)
    ax.set_xlabel(r'$k_1 N_2 + k_2$', fontsize=12)
    ax.set_ylim([-0.1,maxE-minE+0.2])
    return fig, ax

def Plot_spectra(X, Y):
    fig, axs = plt.subplots(figsize=(3,3))
    axs.set_xlabel(r'$\theta_x/2\pi$', fontsize=12)
    axs.tick_params(axis='both', labelsize=12)
    axs.set_ylabel(r'$E-E_{min} (meV)$', fontsize=12)

    for i in range(3):  # Ensure Y has at least 3 columns
        axs.scatter(X, Y[:, i], s=8, c='blue')  # Fixed marker size and color

    return fig, axs

theta = 1.8 #float(sys.argv[1])
Nhole = 9
Vd = 0.0 #float(sys.argv[2]) #displacement field

dtheta, Evals =  main(theta, Nhole, Vd)
fig,ax = Plot_spectra(dtheta, Evals)

fig.savefig('spectralFlow_27sites_nh='+str(Nhole)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'P=0.png',bbox_inches='tight',dpi=300)
fig.savefig('spectralFlow_27sites_nh='+str(Nhole)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'P=0.pdf',bbox_inches='tight',dpi=300)
