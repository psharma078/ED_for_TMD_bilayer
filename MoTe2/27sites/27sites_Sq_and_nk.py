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
from pylanczos import PyLanczos
import time
import sys

num_cores = multiprocessing.cpu_count()

epsilon_r = float(sys.argv[1])  #coulomb parameter
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
    Nk = 1 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3
    N1 = 9
    N2 = 3
    #Nup = 8

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
    
    def compute_F_mat():
        #k12list = np.zeros_like(klist,dtype=np.float64)
        #k12list[:,0] = klist[:,0]/N1
        #k12list[:,1] = klist[:,1]/N2
        #kxylist = k12list @ np.array([Q1,Q2])
        kxylist = mnklist @ np.array([T1, T2])
        #################################################################
        ## Solve for up spin
        solver = lambda k: eigh_sorted(construct_ham(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2  #including layer index
        
        singleVec = np.array([result[jj][1] for jj in range(n_k34)])
        singleVec = singleVec[:,:,-1]

        EEup = np.array([result[jj][0] for jj in range(n_k34)]).reshape([N1,N2,dim],order='F')[:,:,-1] #[:,:,-1] for only the ground state
        print("single EEup:", EEup)
    
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

    def get_k_q_ids(k3_id,k4_id,mnklist,qlist,glist, gtrans):
        k3_q = mnklist[k3_id] + qlist
        k4_q = mnklist[k4_id] - qlist
        g_k3_q, braket_k3_q = compute_reduce_k(k3_q,mnklist, glist, gtrans)
        g_k4_q, braket_k4_q = compute_reduce_k(k4_q, mnklist, glist, gtrans)
        k3_q_id = find_id_in_klist(braket_k3_q)
        k4_q_id = find_id_in_klist(braket_k4_q)
        g_k3_q_id = find_id_in_glist(g_k3_q)
        g_k4_q_id = find_id_in_glist(g_k4_q)
        return k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id
    
    def compute_coulomb(k3_id,k4_id,mnklist,qlist, Fmat1,Fmat2, Vlist, glist, gtrans):
        k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id = get_k_q_ids(k3_id,k4_id,mnklist,qlist,glist, gtrans)
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
         

    def create_table_k1234_from_k3k4(ii,jj,mnklist, qlist, Fmat_1, Fmat_2, Vlist, glist, gtrans):
        k1k2k3k4 = []
        Vk1k2, k1k2 = compute_coulomb(ii,jj,mnklist,qlist,Fmat_1, Fmat_2, Vlist, glist, gtrans)
        for kk in k1k2:
            k1k2k3k4.append(np.concatenate((kk,np.array([ii,jj]))))
        return k1k2k3k4, Vk1k2
    
    def coulomb_matrix_elements(mnklist,qlist, Vlist, Fmat_up):
        k1234_upup = []
        Vc_upup = []
        glist, gtrans = glist_gtrans(Nk,2)
        for ii in range(n_k34):
            for jj in range(ii):
                ##########################
                ### ii neq jj
                _,Vk1234_temp = create_table_k1234_from_k3k4(ii,jj,mnklist,qlist, Fmat_up, Fmat_up, Vlist, glist, gtrans)
                #k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(jj,ii,mnklist,qlist, Fmat_up, Fmat_up, Vlist, glist, gtrans)
                k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                Vc_temp = []
                for ll in range(len(Vk1234)):
                    Vc_temp.append(2*(Vk1234[ll] - Vk1234_temp[ll]))
                Vc_upup.append(Vc_temp)
        return k1234_upup, Vc_upup
    
    def create_coulomb_table(k1234_upup, Vc_upup):
        return np.vstack(k1234_upup), np.hstack(Vc_upup)
    
    qlist, klist = sample_q(N1,N2,Nk, mnklist)
     
    Vlist = coulomb_potential(qlist)
    Fmat_up, EEup, singleVec = compute_F_mat()
    
    k1234_upup, Vc_upup = coulomb_matrix_elements(mnklist,qlist, Vlist, Fmat_up)
    k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)
    
    up_configs = np.array(bitstring_config(n_k34,Nup))
    Ktot, configsGroupID, singleE = groupingKsum(up_configs, mnklist, EEup, N1, N2,Nk)
    
    #solver = lambda gid: manybody_Hamiltonian(gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))
    #Evals = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configsGroupID)
    
    ##Calculate occupation desity (nk)
    #Gs_Ktot_ind = [1]
    Gs_Ktot_ind = [0, 8, 13]#[0,5,7]#[4,9,17]#insert the k index (=k1*N2 + k2)  where gs degeneracy happens
    configGid = []
    configurs = []
    for g in Gs_Ktot_ind:
        configid = configsGroupID[g]
        configGid.append(configid)
        configurs.append(up_configs[configid])

    if len(Gs_Ktot_ind)>1: # If degeneracy happens at diff k points
        solver = lambda gid: manybody_Hamiltonian(len(Gs_Ktot_ind),gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))
        Evecs = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configGid)
    else: # If degeneracy happens at same k point
        Evecs = manybody_Hamiltonian(len(Gs_Ktot_ind),configGid[0], up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))

    solve = lambda kindex: density_nk(kindex, configurs, Evecs)
    nk = Parallel(n_jobs=num_cores)(delayed(solve)(kindex) for kindex in range(len(klist)))

    #charge structure factor
    solver = lambda q: CSF_for_degenrateGS(q, mnklist, Evecs, configurs, singleVec, klist_lookup)
    Sq =  np.array(Parallel(n_jobs=num_cores)(delayed(solver)(q) for q in klist))
    Sq[0] = Sq[0]-Nup*Nup
    Sq = Sq/(N1*N2)
    for i in range(len(klist)):
        print(klist[i], Sq[i], nk[i])
    Sq = Sq/max(Sq)
    
    #Ktot_id = np.array([klist_lookup[tuple(k)] for k in Ktot])
    #Ktot = klist[Ktot_id]
    
    #hexagonalBZmap(mnklist, T1, T2)
    #sys.exit()
    return mnklist, Sq, nk, T1, T2

def CSF(qval, klist, evec, config, Len, singleVec, klist_lookup):
    config_lookup = {value: index for index, value in enumerate(config)}
    k_minus_q = klist - qval
    glist, gtrans = glist_gtrans(1,2)
    _,k_minus_q_mBZ = compute_reduce_k(k_minus_q, klist, glist, gtrans)
    k_minus_q_tag = np.array([klist_lookup[tuple(kmq)] for kmq in k_minus_q_mBZ])

    k_plus_q = klist + qval
    _,k_plus_q_mBZ = compute_reduce_k(k_plus_q, klist, glist, gtrans)
    k_plus_q_tag =  np.array([klist_lookup[tuple(kpq)] for kpq in k_plus_q_mBZ])

    S_q = 0
    for kprime in range(len(klist)):
        config_kprime = getspin(config,kprime)
        valid_config_kprimeID = (config_kprime==1).nonzero()[0]
        f1, newconfig = c_an(config[valid_config_kprimeID], kprime)
        valid_config_kmqID = (getspin(newconfig, k_minus_q_tag[kprime])==0).nonzero()[0]
        f2, newconfig = c_dag(newconfig[valid_config_kmqID], k_minus_q_tag[kprime])
        fsign1 = f1[valid_config_kmqID] * f2
        valid_ID = valid_config_kprimeID[valid_config_kmqID]
        Lambda_kmq = sum(singleVec[k_minus_q_tag[kprime]].conjugate() * singleVec[kprime])
        #print(kprime, k_minus_q_tag[kprime], valid_ID, [bin(nc)[2:].zfill(N1*N2) for nc in newconfig])
        for k in range(len(klist)):
            config_k =  getspin(newconfig,k)
            valid_config_kID = (config_k==1).nonzero()[0]
            f3, new_config = c_an(newconfig[valid_config_kID], k)
            f3 = fsign1[valid_config_kID]*f3
            valid_config_kqID = (getspin(new_config, k_plus_q_tag[k])==0).nonzero()[0]
            f4, new_config = c_dag(new_config[valid_config_kqID], k_plus_q_tag[k])
            fsign = f3[valid_config_kqID] * f4
            orginal_idlist = valid_ID[valid_config_kID[valid_config_kqID]]
            newconfig_id = np.array([config_lookup[nc] for nc in new_config]).astype(int)
            Lambda_kpq = sum(singleVec[k_plus_q_tag[k]].conjugate() * singleVec[k])
            if Len == 1:
                for j in range(3):
                    S_q += Lambda_kpq*Lambda_kmq * sum((evec[:,j])[newconfig_id].flatten().conjugate() * (evec[:,j])[orginal_idlist].flatten() * fsign)
            else:
                S_q += Lambda_kpq*Lambda_kmq * sum(evec[newconfig_id].flatten().conjugate() * evec[orginal_idlist].flatten() * fsign)

    return S_q

def CSF_for_degenrateGS(qval, klist, evecs, configs, singleVec, klist_lookup):
    Len = len(configs)
    Sq = 0
    if Len==1:
        Sq += CSF(qval, klist, evecs, configs[0], Len, singleVec, klist_lookup)
    else:
        for p in range(Len):
            Sq += CSF(qval, klist, evecs[p], configs[p], Len, singleVec, klist_lookup)
    return Sq.real/3.0

def density_nk(kidx, configs, evecs):
    nk = 0
    nstate = 0
    Len = len(configs)
    if Len > 1: # If degeneracy happens at different k-points
        for i, evec in enumerate(evecs):
            nk_each = sum(evec.flatten().conjugate() * getspin(configs[i],kidx) * evec.flatten())
            nk += nk_each.real
            nstate += 1
    else:# If degeneracy happens at same k-points
        for i in range(3):
            nk += sum(evecs[:,i].flatten().conjugate() * getspin(configs[0],kidx) * evecs[:,i].flatten()).real
            nstate += 1

    nk = nk/nstate
    return nk

def boundary(k):
    ll1 = np.array([1,sqrt(3)])
    ll2 = np.array([-1,sqrt(3)])
    logic = False
    if (k[0]<=0.5) & (k@ll1<=1) & (k@ll2<=1):
        logic = True
    return logic

def hexagonalBZmap(Ktot, T1, T2):
    g1 = np.array([1.0,0])
    g2 = np.array([-1.0/2, sqrt(3)/2])
    k1 = (2*g1+g2)/3
    k2 = (g1-g2)/3
    k3 = (-2*g2-g1)/3
    k4 = -k1
    k5 = -k2
    k6 = -k3
    kk = np.array([k1,k2,k3,k4,k5,k6,k1])
    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot([kk[i][0],kk[i+1][0]],[kk[i][1],kk[i+1][1]], color='k')
    '''
    NN = [np.array([0,0]),g1, g1+g2, g2, np.array([0,0])]
    for i in range(4):
        ax.plot([NN[i][0],NN[i+1][0]],[NN[i][1],NN[i+1][1]], color='r')
    '''
    g1norm = np.linalg.norm(np.array([6,-3]) @ np.array([T1,T2]))
    newK = (Ktot @ np.array([T1,T2]))/g1norm
    #ax.scatter(newK[:,0], newK[:,1],color='red')
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
        #print(Ktot[i], k)
        #i += 1
    KK = np.array(KK)
    #ax.scatter(KK[:,0], KK[:,1],color='blue')
    ax.set_aspect('equal')
    ax.axis('off')
    #plt.show()
    return KK, fig, ax

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
    x_braket = qi[gindex].flatten()
    if len(x_braket)==0:
        print("empty value")
        import sys
        sys.exit
    #print(x_braket)
    g_q = glist[gindex].flatten()
    return g_q, x_braket

def compute_reduce_k(q_list, mnklist, glist, gtrans):
    #p1 = np.array([6,-3])
    #p2 = np.array([-3,6])
    #glist = np.arange(-Nc*Nmax,Nc*Nmax+1,dtype=np.int32)
    #glist = np.array([np.array([glist[ii],glist[jj]]) for jj in range(2*Nc*Nmax+1) for ii in range(2*Nc*Nmax+1)], dtype=np.int32)
    #gtrans = glist@np.array([p1,p2])
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
    VV = np.array(Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)).reshape([N1,N2,dim,dim],order='F')
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

def groupingKsum(configs, klist, EEup, N1, N2,Nk):
    dimH = len(configs)
    binary_lists = np.fliplr(config_array(configs,N1*N2)) #convert binary num to array
    occ_ksum = binary_lists@klist
    glist, gtrans = glist_gtrans(Nk, 1)
    _,occ_ksum = compute_reduce_k(np.array(occ_ksum), klist, glist, gtrans)
    occ_ksum += np.array([2,2])
    Ktot, grouped_k = two_index_unique(occ_ksum,N1*N2)
    Ktot -= np.array([2,2])
    singleE = binary_lists @ EEup.flatten(order='F')
    return Ktot,  grouped_k, singleE


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
    
    ##Eigs = eigsh(matrix,k=6,which='SA',sigma=None,return_eigenvectors=False)
    ##Eigs = np.sort(Eigs.real)
    if Len > 1:
        Eigs, vecs = eigsh(matrix,k=1,which='SA',sigma=None,return_eigenvectors=True)
    else:
        Eigs,vecs = eigsh(matrix,k=3,which='SA',sigma=None,return_eigenvectors=True)
    print(Eigs)
    return vecs

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

def plot_ED_energies(Ktot, Evals):
    num_K = Ktot.shape[0]
    Evals_array = np.array(Evals)
    print("ktot,         Eigen values (meV)")
    for i in range(len(Ktot)):
        print(Ktot[i], Evals[i])
    minE = Evals_array.min()
    maxE = Evals_array.max()
    print()
    print("Eigenvalues sorted")
    print(np.sort(Evals_array.flatten()))
    #print(Evals_array - minE)
    fig, ax = plt.subplots()
    num_K = 3*Ktot[:,0] + Ktot[:,1]
    ax.plot(np.arange(num_K), (Evals_array - minE),'k.',markersize=10)
    ax.set_ylabel(r'$E - E_{GS}(meV)$')
    ax.set_xlabel(r'$k_1 N_2 + k_2$')
    ax.set_ylim([-0.1,maxE-minE+0.2])
    return fig, ax

def plot_nk_in_hBz(T1,T2, ktot, nk, Label):
    kBz, fig, ax = hexagonalBZmap(ktot, T1,T2)
    cm = plt.cm.get_cmap('turbo')
    #ax.scatter(kBz[:,0], kBz[:,1],s=abs(connected_Sq)*1000, color='r', marker='h',edgecolor='black')
    sc=ax.scatter(kBz[:,0], kBz[:,1],c=nk, vmin=0, vmax=1, s=200,cmap=cm, marker='h', alpha=0.8)
    plt.colorbar(sc, fraction=0.02, pad=0.004, label=Label)
    return fig, ax

theta = float(sys.argv[2])
Nhole = 2
Vd = float(sys.argv[3]) #displacement field
ktot, Sq, nk, T1, T2 =  main(theta,Nhole,Vd)

fig_nk, ax_nk = plot_nk_in_hBz(T1,T2, ktot, nk, r'$n_k$')
fig, ax = plot_nk_in_hBz(T1,T2, ktot, Sq, r'$S(q)/S_o$')

#fig_nk.savefig('nk_in_mBZ_'+str(N1)+'x'+str(N2)+'_nh='+str(Nup)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'_Vd='+str(round(Vd,2))+'.png',bbox_inches='tight',dpi=300)
#fig.savefig('Sq_'+str(N1)+'x'+str(N2)+'_nh='+str(Nup)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'_Vd='+str(round(Vd,2))+'.png',bbox_inches='tight',dpi=300)
plt.show()

#folder = "half_fill/"
#import os
#fig_yang.savefig(os.path.join(folder, f"ED_6x3_nh={str(Nhole)}_ep={str(epsilon_r)}_theta={str(round(theta,2))}_Vd={str(Vd)}.png"), bbox_inches='tight', dpi=300)
plt.show()
