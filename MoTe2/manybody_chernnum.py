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

def main(theta, N1, N2, Nup, Vd):
    theta = theta/180*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    kappa_p = (2 * Q1 + Q2)/3
    kappa_m = (Q1 - Q2)/3
    mpt=  Q1/2

    Nk = 6 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3

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

    def gen_layer_hamiltonian(k, delk):
        me = 9.10938356e-31 # kg
        m = 0.62*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV
        kxlist_top = Qxlist + (k[0] + delk[0]) - kappa_p[0]
        kylist_top = Qylist + (k[1] + delk[1]) - kappa_p[1]
        kxlist_bottom = Qxlist + (k[0] + delk[0]) - kappa_m[0]
        kylist_bottom = Qylist + (k[1] + delk[1]) - kappa_m[1]
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
        ## w = -18.0 # meV
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

    def compute_F_mat(klist, dk):
        k12list = np.zeros_like(klist,dtype=np.float64)
        k12list[:,0] = klist[:,0]/N1
        k12list[:,1] = klist[:,1]/N2
        kxylist = k12list @ np.array([Q1,Q2])
        #################################################################
        ## Solve for up spin
        solver = lambda k: eigh_sorted(construct_ham(k, dk))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2  #including layer index

        singleVec = np.array([result[jj][1] for jj in range(n_k34)])
        singleVec = singleVec[:,:,-1]

        EEup = np.array([result[jj][0] for jj in range(n_k34)]).reshape([N1,N2,dim],order='F')[:,:,-1] #[:,:,-1] for only the ground state
        ##print("single EEup:", EEup)

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

    @njit
    def find_id_in_klist(braket_k):
        return braket_k[:,0]+N1*braket_k[:,1]
    @njit
    def find_id_in_glist(g_k):
        g_k += Nk
        return g_k[:,0]+(2*Nk+1)*g_k[:,1]

    #################################
    ## CHECK INDICES

    @njit
    def get_k_q_ids(k3_id,k4_id,klist,qlist):
        k3_q = klist[k3_id] + qlist
        k4_q = klist[k4_id] - qlist
        g_k3_q, braket_k3_q = compute_reduce_k(k3_q,N1,N2)
        g_k4_q, braket_k4_q = compute_reduce_k(k4_q,N1,N2)
        k3_q_id = find_id_in_klist(braket_k3_q)
        k4_q_id = find_id_in_klist(braket_k4_q)
        g_k3_q_id = find_id_in_glist(g_k3_q)
        g_k4_q_id = find_id_in_glist(g_k4_q)
        return k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id

    @njit
    def compute_coulomb(k3_id,k4_id,klist,qlist,Fmat1,Fmat2, Vlist):
        k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id = get_k_q_ids(k3_id,k4_id,klist,qlist)
        ind_k = (g_k3_q_id<n_g) & (g_k3_q_id>=0) & (g_k4_q_id<n_g) & (g_k4_q_id>=0) & (k3_q_id < k4_q_id)
        k3_q_id = k3_q_id[ind_k]
        k4_q_id = k4_q_id[ind_k]
        V = Fmat1[g_k3_q_id[ind_k]*n_k34**2 + k3_id*n_k34 + k3_q_id] * Fmat2[g_k4_q_id[ind_k]*n_k34**2 + k4_id*n_k34 + k4_q_id] * Vlist[ind_k]
        # V = Fmat1[k3_q_id*n_g*n_k34 + k3_id*n_g + g_k3_q_id[ind_k]] * Fmat2[k4_q_id*n_g*n_k34 + k4_id*n_g + g_k4_q_id[ind_k]] * Vlist[ind_k] #
        k1k2list = np.vstack((k3_q_id,k4_q_id)).T
        k1k2, k1k2_ids =  two_index_unique(k1k2list,n_k34)
        Vk1k2 = List()
        for k_id in k1k2_ids:
            Vk1k2.append(np.sum(V[k_id]))
        return Vk1k2, k1k2

    def coulomb_potential(qlist):
        zeroidx = np.nonzero(np.prod(qlist==0,axis=1))[0][0]
        qxy = np.zeros_like(qlist,dtype=np.float64)
        qxy[:,0] = qlist[:,0]/N1
        qxy[:,1] = qlist[:,1]/N2
        qxy = qxy @ np.array([Q1,Q2]) # in m^-1
        absqlist = np.linalg.norm(qxy,axis=1)

        k0 = 8.99e9
        J_to_meV = 6.242e21  #meV
        e_charge = 1.602e-19 # coulomb
        Area = np.sqrt(3)/2*N1*N2*aM**2
        Vc = 2*np.pi*e_charge**2/(epsilon_r*absqlist)*J_to_meV/Area*k0

        Vc[zeroidx]=0
        return Vc

    @njit
    def create_table_k1234_from_k3k4(ii,jj,klist, qlist, Fmat_1, Fmat_2, Vlist):
        k1k2k3k4 = List()
        Vk1k2, k1k2 = compute_coulomb(ii,jj,klist,qlist,Fmat_1, Fmat_2, Vlist)
        for kk in k1k2:
            k1k2k3k4.append(np.concatenate((kk,np.array([ii,jj]))))
        return k1k2k3k4, Vk1k2
    
    @njit
    def coulomb_matrix_elements(qlist, Vlist, Fmat_up):
        k1234_upup = List()
        Vc_upup = List()
        for ii in range(n_k34):
            for jj in range(ii):
                ##########################
                ### ii neq jj
                _,Vk1234_temp = create_table_k1234_from_k3k4(ii,jj,klist,qlist,Fmat_up, Fmat_up, Vlist)
                #k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(jj,ii,klist,qlist,Fmat_up, Fmat_up, Vlist)
                k1234_upup.append(k1k2k3k4)
                #Vc_upup.append(Vk1234)
                Vc_temp = List()
                for ll in range(len(Vk1234)):
                    Vc_temp.append(2*(Vk1234[ll] - Vk1234_temp[ll]))
                Vc_upup.append(Vc_temp)

        return k1234_upup, Vc_upup

    def create_coulomb_table(k1234_upup, Vc_upup):
        return np.vstack(k1234_upup), np.hstack(Vc_upup)

    def diagonalization(klist, qlist, Vlist, binary_lists, configGID, dk):
        Fmat_up, EEup, singleVec = compute_F_mat(klist, dk)
        singleE = binary_lists @ EEup.flatten(order='F')
        k1234_upup, Vc_upup = coulomb_matrix_elements(qlist, Vlist, Fmat_up)
        k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)
        Evecs = []
        Len = len(configGID)
        for gid in configGID:
            Evecs.append(manybody_Hamiltonian(Len, gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128)))
        if Len==1: Evecs = Evecs[0]
        return Evecs, singleVec

    klist, qlist = sample_k_q(N1,N2,Nk)
    Vlist = coulomb_potential(qlist)

    up_configs = np.array(bitstring_config(n_k34,Nup))
    binary_lists = np.fliplr(config_array(up_configs,N1*N2))
    Ktot, configsGroupID = groupingKsum(binary_lists, klist, N1, N2)

    #constructing dk grids
    dTheta = np.linspace(0, 1, 14)#change
    dTheta_grid = np.array([np.array([th1, th2]) for th1 in dTheta for th2 in dTheta])
    #print(dTheta_grid)
    delta_klist = (dTheta_grid/np.array([N1,N2])) @ np.array([Q1, Q2])

    Gs_Ktot_ind = [0]#[0,3,6,9,12]#[0, 8, 13]##insert the k index (=k1*N2 + k2)  where gs degeneracy occur
    configGID = []
    configurs = []
    for g in Gs_Ktot_ind:
        configid = configsGroupID[g]
        configGID.append(configid)
        configurs.append(up_configs[configid])

    #Vvec, Uvec = [], []
    #for dk in delta_klist:
    #   mvec, svec = diagonalization(klist, qlist, Vlist, binary_lists, configGID, dk)
    #   Vvec.append(mvec)
    #   Uvec.append(svec)
    solver = lambda dk: diagonalization(klist, qlist, Vlist, binary_lists, configGID, dk)
    results = Parallel(n_jobs=num_cores)(delayed(solver)(dk) for dk in delta_klist)

    Vvec = [results[i][0] for i in range(len(delta_klist))]
    Uvec = np.array([results[i][1] for i in range(len(delta_klist))])

    numstate = len(Vvec[0])
    results = 0
    Vecs = []
    for j in range(numstate):
        Vecs.append(np.array([Vvec[i][j] for i in range(len(delta_klist))]))
    del Vvec

    bin_list_gs_configs = [binary_lists[configsGroupID[ind]] for ind in Gs_Ktot_ind]
    if len(Gs_Ktot_ind)==1:
        solver = lambda jj: manybody_chernnum(bin_list_gs_configs[0], Vecs[jj],Uvec, N1*N2, Nup, n_g*2)
    else:
        solver = lambda jj: manybody_chernnum(bin_list_gs_configs[jj], Vecs[jj],Uvec, N1*N2, Nup, n_g*2)
    results = Parallel(n_jobs=num_cores)(delayed(solver)(jj) for jj in range(numstate))

    ##cn=manybody_chernnum(bin_list_gs_configs[0], Vecs[0],Uvec, N1*N2, Nup, n_g*2)
    manybody_cn = np.array([results[i][1] for i in range(numstate)])
    F12 = np.array([results[i][0] for i in range(numstate)])*2*pi
    print('Ktot = ',Ktot[Gs_Ktot_ind])
    print('manybody Chern No. = ',manybody_cn)
    print('sum of cn =', sum(manybody_cn))
    print()
    return F12, dTheta_grid, Ktot[Gs_Ktot_ind]

def manybody_chernnum(bin_list, Vec, Uvec, N, Np, ng):
    #occ_BlochState = np.array([(bin_list[i]==1).nonzero()[0] for i in range(len(bin_list))])
    occ_BlochState = (bin_list.flatten()==1).nonzero()[0].reshape(len(bin_list),Np) % N
    delk_len = len(Uvec)
    len_delk1 = int(np.sqrt(delk_len))
    Vec = Vec.reshape(len_delk1, len_delk1, len(bin_list))
    Vx = Vec.conjugate()*np.roll(Vec,-1,axis=0)
    Vy = Vec.conjugate()*np.roll(Vec,-1,axis=1)
    del Vec
    Ux = 0
    Uy = 0
    for l in range(len(occ_BlochState)):
        U_vec = np.array([Uvec[i][occ_BlochState[l]] for i in range(delk_len)]).reshape(len_delk1,len_delk1,Np,ng)
        Ux += Vx[:,:,l] * np.product(np.sum(U_vec.conjugate() * np.roll(U_vec, -1, axis=0), axis=3), axis=2)
        Uy += Vy[:,:,l] * np.product(np.sum(U_vec.conjugate() * np.roll(U_vec, -1, axis=1), axis=3), axis=2)

    Ux = Ux/np.abs(Ux)
    Uy = Uy/np.abs(Uy)
    F12 = np.log(Ux * np.roll(Uy,-1,axis=0) / (Uy * np.roll(Ux,-1,axis=1)))[:-1,:-1]
    F12 = np.imag(F12)/(2*np.pi)
    #print(F12)
    mbChernNum = np.sum(F12,axis=(0,1))
    return F12, mbChernNum

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

@njit
def compute_reduce_k(k12,N1,N2):
    gx, x_braket = np.divmod(k12,np.array([N1,N2]))
    return gx, x_braket

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

def groupingKsum(binary_lists, klist, N1, N2):
    #binary_lists = np.fliplr(config_array(configs,N1*N2)) #convert binary num to array
    occ_ksum = binary_lists@klist
    _,occ_ksum = compute_reduce_k(np.array(occ_ksum),N1,N2)
    Ktot, grouped_k = two_index_unique(occ_ksum,N2)
    #singleE = binary_lists @ EEup.flatten(order='F')
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

def manybody_Hamiltonian(Len, configs_indx, up_configs, singleE, k1234, V1234):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))
    Vecs = []
    Eigs = []

    if Len==1:
        Eigs, vecs = eigsh(matrix,k=3,which='SA',sigma=None,return_eigenvectors=True)
        idx = Eigs.argsort()
        Eigs = Eigs[idx]
        Vecs = [vecs[:,ID] for ID in idx]
    elif Len>1:
        Eigs, Vecs = eigsh(matrix,k=3,which='SA',sigma=None,return_eigenvectors=True)
        #idx = Eigs.argsort()
        #Eigs = Eigs[idx]
        #Vecs = Vecs[:,idx[2]]
        Vecs = Vecs.flatten()

    #print(Eigs.real)
    return Vecs

@njit
def convert_to_1_id(k1k2list, N):
# with C ordering
    return k1k2list[:,0]*N + k1k2list[:,1]

@njit
def convert_to_2_id(klist, N):
# with C ordering
    x,y = np.divmod(klist,N)
    return np.vstack((x,y)).T

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

def colorPlot(X, Y, Z):
    #fig, ax = plt.subplots(1,3,figsize=(8.7,2.3), sharey=True)##for 3 fold degeneracy
    Len = len(Z)
    fig, ax = plt.subplots(1,Len,figsize=(10,1.7), sharey=True) ##for 5 fold degeneracy
    color = 'jet'
    for i in range(Len-1):
        c = ax[i].pcolor(X, Y, Z[i], cmap=color, vmin=Z.min(), vmax=Z.max(),alpha=0.75)
        ax[i].set_xlabel(r'$\theta_1/2\pi$',fontsize=12)
    c = ax[Len-1].pcolor(X, Y, Z[Len-1], cmap=color, vmin=Z.min(), vmax=Z.max(), alpha=0.75)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.subplots_adjust(left=0.1, right=0.94, wspace=0.1)
    cb = plt.colorbar(c,ax=ax,pad=0.007)
    cb.set_label(r'$F(\theta_1,\theta_2)$', fontsize=11)
    ax[Len-1].set_xlabel(r'$\theta_1/2\pi$', fontsize=12)
    ax[0].set_ylabel(r'$\theta_2/2\pi$',fontsize=12)
    ##ax.plot(11.2, 13.3, 'k.', marker='*', markersize=10)
    return fig, ax

theta = float(sys.argv[2])
Nx, Ny, Nhole = 6, 3, 12
Vd = float(sys.argv[3]) #displacement field
F12, dtheta, ktot =  main(theta,Nx, Ny, Nhole, Vd)
print(F12)
dtheta1 = dtheta[:,0].reshape(len(F12[0])+1,len(F12[0])+1)[:-1,:-1]
dtheta2 = dtheta[:,1].reshape(len(F12[0])+1,len(F12[0])+1)[:-1,:-1]

#fig, ax = colorPlot(dtheta1,dtheta2,F12)
#fig.savefig('manybody_BerryCurvature_N1='+str(Nx)+'_N2='+str(Ny)+'_Nh='+str(Nhole)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'_Vd='+str(Vd)+'.png',bbox_inches='tight',dpi=300)
#fig.savefig('manybody_BerryCurvature_N1='+str(Nx)+'_N2='+str(Ny)+'_Nh='+str(Nhole)+'_ep='+str(epsilon_r)+'_theta='+str(round(theta,2))+'_Vd='+str(Vd)+'.pdf',bbox_inches='tight',dpi=300)

plt.show()
