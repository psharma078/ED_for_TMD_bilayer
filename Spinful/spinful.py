import numpy as np
import scipy as sp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import sin,cos,exp, pi, sqrt
from joblib import Parallel, delayed
import multiprocessing
from numba import jit,njit
from numba.typed import List
import time
from interaction import *
from operator import itemgetter
#from pylanczos import PyLanczos

##### THERE IS A 0.5 differences in coulomb matrix elements.


num_cores = multiprocessing.cpu_count()

def main(theta, N1, N2, Nup, Ndn, Vd):
    theta = theta/180*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    kappa_p = (2 * Q1 + Q2)/3  # with g1,g3
    kappa_m = (Q1 - Q2)/3 # with g1, g3  
    mpt=  Q1/2
    Nk = 6 #cutoff number [-Nk:Nk]*Q2 + [-Nk:Nk]*Q3
    #N1 = 4
    #N2 = 4
    n_k34 = N1*N2
    #Nup = 2
    #Ndn = 2
    n_g = (2*Nk+1)**2
    n_q =  n_g * n_k34
    Nklist = np.arange(-Nk,Nk+1)
    Nkoneslist = np.ones(2*Nk+1)
    nmlist = [[n,m] for m in Nklist for n in Nklist]
    Qlist = np.array(nmlist)@np.array([Q1,Q2])
    
    diagm = np.diag(np.ones(2*Nk,dtype=np.complex128),-1)
    diagp = diagm.T
    diag_Nk= np.eye(2*Nk+1,dtype=np.complex128)
    delta_nn_mmp1 = np.kron(diagm, diag_Nk)  # Q, Q+Q2 ; with g1, g3
    # delta_nn_mmp1 = np.kron(diagm, diagp) #Q, Q+Q2'-Q1' ; with g1, g2
    delta_nnp1_mm = np.kron(diag_Nk,diagm) # Q, Q+Q1; 
    delta_nnm1_mmm1 = np.kron(diagp,diagp) # Q, Q-Q1 - Q2; with g1, g3
    # delta_nnm1_mmm1 = np.kron(diagp, diag_Nk)  # Q, Q - Q2 ; with g1, g2
    delta_nn_mm = np.kron(diag_Nk,diag_Nk)
    delta_nn_mmm1 = delta_nn_mmp1.T

    halfdim = (Nk*2+1)**2
    ham_dim = 2*halfdim
    ### kline Gamma -- M ---K---Gamma
    def special_kline(npts):
        Gammapt = np.array([0.,0.])
        mpt = Q1/2
        kline1 = klines(Gammapt,mpt,npts)
        kline2 = klines(mpt,kappa_p,npts)
        kline3 = klines(kappa_p,Gammapt,npts)
        klabels = [r"$\gamma$", r'$m$', r'$\kappa_+$', r'$\gamma$']
        kcoord = np.array([0,(npts-1),2*(npts-1),3*(npts-1)+1])
        return np.concatenate((kline1[:-1,:],kline2[:-1,:],kline3[:,:])), kcoord, klabels

    def gen_layer_hamiltonian(k):
        # hear k is in (kx,ky) not k1,k2
        me = 9.10938356e-31 # kg
        m = 0.62*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV

        kxlist_top = Qlist[:,0] + k[0] - kappa_p[0]
        kylist_top = Qlist[:,1] + k[1] - kappa_p[1]
        kxlist_bottom = Qlist[:,0] + k[0] - kappa_m[0]
        kylist_bottom = Qlist[:,1] + k[1] - kappa_m[1]
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
        ham[halfdim:,:halfdim] = Delta_T
        return ham

    def gen_layer_hamiltonian_dn(k):
        # hear k is in (kx,ky) not k1,k2
        me = 9.10938356e-31 # kg
        m = 0.62*me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21  #meV
        prefactor = hbar**2/(2*m)*J_to_meV

        kxlist_top = Qlist[:,0] + k[0] + kappa_p[0]
        kylist_top = Qlist[:,1] + k[1] + kappa_p[1]
        kxlist_bottom = Qlist[:,0] + k[0] + kappa_m[0]
        kylist_bottom = Qlist[:,1] + k[1] + kappa_m[1]
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


    def construct_ham_dn(k):
        H_top,H_bottom = gen_layer_hamiltonian_dn(k)
        Delta_T = gen_tunneling()
        ham = np.zeros((ham_dim,ham_dim),dtype=np.complex128)
        ham[:halfdim,:halfdim] = H_top
        ham[halfdim:,halfdim:] = H_bottom
        ham[:halfdim,halfdim:] = Delta_T
        ham[halfdim:,:halfdim] = Delta_T.conj().T
        return ham 

    def compute_F_mat(klist):
        k12list = np.zeros_like(klist,dtype=np.float64)
        k12list[:,0] = klist[:,0]/N1
        k12list[:,1] = klist[:,1]/N2
        kxylist = k12list @ np.vstack((Q1,Q2))
        #################################################################
        ## Solve for up spin
        solver = lambda k: eigh_sorted(construct_ham(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2
        EEup = np.array([result[jj][0] for jj in range(n_k34)]).reshape([N1,N2,dim],order='F')[:,:,-1]
        VV = np.array([result[jj][1][:,-1] for jj in range(n_k34)], dtype=np.complex128).reshape([N1,N2, 2*Nk+1, 2*Nk+1,2],order='F')
        Fmat_up = np.zeros([N1,N2,N1,N2,2*Nk+1,2*Nk+1], dtype=np.complex128)
        Fmat_up[:,:,:,:,Nk,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:,:].conj(), VV[:,:,:,:,:])
        for jj in range(1,Nk+1):
            for ii in range(1,Nk+1):
                Fmat_up[:,:,:,:,Nk+ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,jj:,:].conj(), VV[:,:,:(-ii),:(-jj),:])
                Fmat_up[:,:,:,:,Nk+ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,:(-jj):,:].conj(), VV[:,:,:(-ii),jj:,:])
                Fmat_up[:,:,:,:,Nk-ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),jj:,:].conj(), VV[:,:,ii:,:(-jj),:])
                Fmat_up[:,:,:,:,Nk-ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),:(-jj),:].conj(), VV[:,:,ii:,jj:,:])
        for jj in range(1,Nk+1):
            Fmat_up[:,:,:,:,Nk+jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,jj:,:,:].conj(), VV[:,:,:(-jj),:,:])
            Fmat_up[:,:,:,:,Nk-jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-jj),:,:].conj(), VV[:,:,jj:,:,:])
            Fmat_up[:,:,:,:,Nk,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,jj:,:].conj(), VV[:,:,:,:(-jj),:])
            Fmat_up[:,:,:,:,Nk,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:(-jj),:].conj(), VV[:,:,:,jj:,:])
        #################################################################
        ## Solve for down spin
        solver = lambda k: eigh_sorted(construct_ham_dn(k))
        result = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in kxylist)
        dim = n_g *2
        EEdn = np.array([result[jj][0] for jj in range(n_k34)]).reshape([N1,N2,dim],order='F')[:,:,-1]
        VV = np.array([result[jj][1][:,-1] for jj in range(n_k34)], dtype=np.complex128).reshape([N1,N2, 2*Nk+1, 2*Nk+1, 2],order='F')       
        Fmat_dn = np.zeros([N1,N2,N1,N2,2*Nk+1,2*Nk+1], dtype=np.complex128)
        Fmat_dn[:,:,:,:,Nk,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:,:].conj(), VV[:,:,:,:,:])
        for jj in range(1,Nk+1):
            for ii in range(1,Nk+1):
                Fmat_dn[:,:,:,:,Nk+ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,jj:,:].conj(), VV[:,:,:(-ii),:(-jj),:])
                Fmat_dn[:,:,:,:,Nk+ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,:(-jj):,:].conj(), VV[:,:,:(-ii),jj:,:])
                Fmat_dn[:,:,:,:,Nk-ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),jj:,:].conj(), VV[:,:,ii:,:(-jj),:])
                Fmat_dn[:,:,:,:,Nk-ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),:(-jj),:].conj(), VV[:,:,ii:,jj:,:])
        for jj in range(1,Nk+1):
            Fmat_dn[:,:,:,:,Nk+jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,jj:,:,:].conj(), VV[:,:,:(-jj),:,:])
            Fmat_dn[:,:,:,:,Nk-jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-jj),:,:].conj(), VV[:,:,jj:,:,:])
            Fmat_dn[:,:,:,:,Nk,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,jj:,:].conj(), VV[:,:,:,:(-jj),:])
            Fmat_dn[:,:,:,:,Nk,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:(-jj),:].conj(), VV[:,:,:,jj:,:])
        return Fmat_up.flatten(order='F'), Fmat_dn.flatten(order='F'), EEup, EEdn
    
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
    def compute_coulomb(k3_id,k4_id,klist,qlist,Fmat1,Fmat2, Vlist, same_spin):
        k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id = get_k_q_ids(k3_id,k4_id,klist,qlist)
        ind_k = (g_k3_q_id<n_g) & (g_k3_q_id>=0) & (g_k4_q_id<n_g) & (g_k4_q_id>=0)
        if (same_spin == True):
            ind_k = ind_k & (k3_q_id < k4_q_id)
            k3_q_id = k3_q_id[ind_k]
            k4_q_id = k4_q_id[ind_k]
            V = Fmat1[g_k3_q_id[ind_k]*n_k34**2 + k3_id*n_k34 + k3_q_id] * Fmat2[g_k4_q_id[ind_k]*n_k34**2 + k4_id*n_k34 + k4_q_id] * Vlist[ind_k]
        else:
            k3_q_id = k3_q_id[ind_k]
            k4_q_id = k4_q_id[ind_k]
            V = 0.5 * Fmat1[g_k3_q_id[ind_k]*n_k34**2 + k3_id*n_k34 + k3_q_id] * Fmat2[g_k4_q_id[ind_k]*n_k34**2 + k4_id*n_k34 + k4_q_id] * Vlist[ind_k]
            
        k1k2list = np.vstack((k3_q_id,k4_q_id)).T
        k1k2, k1k2_ids =  two_index_unique(k1k2list,n_k34)
        Vk1k2 = List()
        for k_id in k1k2_ids:
            Vk1k2.append(np.sum(V[k_id]))
        return Vk1k2, k1k2

    
    @njit
    def coulomb_potential(qlist):
        zeroidx = np.where(np.sum(np.abs(qlist),axis=1)==0)[0][0]
        qxy = np.zeros_like(qlist,dtype=np.float64)
        qxy[:,0] = qlist[:,0]/N1
        qxy[:,1] = qlist[:,1]/N2
        qxy = qxy@np.vstack((Q1,Q2)) # in m^-1
        absqlist = np.sqrt(qxy[:,0]**2 + qxy[:,1]**2)
        k0 = 8.99e9
        J_to_meV = 6.242e21  #meV
        e_charge = 1.602e-19 # coulomb
        epsilon = 8.854e-12
        epsilon_r = 10.0
        Area = np.sqrt(3)/2*N1*N2*aM**2
        Vc = 2*np.pi*e_charge**2/(epsilon_r*absqlist)*J_to_meV/Area*k0
        Vc[zeroidx]=0
        return Vc
        

    def bandstructure(energy_range=None):
        npts=40
        klist,kcoord,klabels = special_kline(npts)
        ham_morie_list = []
        for k in klist:
            ham_morie_list.append(construct_ham(k))
        EElist = find_eigenvalues(ham_morie_list)
        fig,ax = plot_bands(EElist,klist,kcoord,klabels,energy_range)
        return fig,ax


    def electron_density(nbz, band_id):
        klist = bz_sampling(nbz,Q1,Q2)
        solver = lambda k: eigh_sorted(construct_ham(k))[1]
        EVlist = Parallel(n_jobs=num_cores)(delayed(solver)(k) for k in klist)
        nx = 50
        ny = 50
        xlist = np.linspace(-200,200,nx)
        ylist = np.linspace(-200,200,ny)
        density = np.zeros((nx,ny))
        xx,yy = np.meshgrid(xlist,ylist)
        x = xx.flatten()
        y = yy.flatten()
        for EV in EVlist:
            wft = np.reshape(np.dot(EV[:halfdim,band_id],exp(1j*(Qxlist[:,np.newaxis]*x[np.newaxis,:] + Qylist[:,np.newaxis]*y[np.newaxis,:]))),(nx,ny))
            wfb = np.reshape(np.dot(EV[halfdim:,band_id],exp(1j*(Qxlist[:,np.newaxis]*x[np.newaxis,:] + Qylist[:,np.newaxis]*y[np.newaxis,:]))),(nx,ny))
            density += np.abs(wft)**2
            density += np.abs(wfb)**2
#             density = densit

        cmap = 'RdBu'
        fig,ax = plt.subplots()
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title('Top Layer')
        c = ax.pcolormesh(xx,yy,density,shading = 'gouraud',cmap = cmap)
        return fig,ax

    @njit
    def create_table_k1234_from_k3k4(ii,jj,klist, qlist, Fmat_1, Fmat_2, Vlist, same_spin=True):
        k1k2k3k4 = List()
        Vk1k2, k1k2 = compute_coulomb(ii,jj,klist,qlist,Fmat_1, Fmat_2, Vlist, same_spin)
        for kk in k1k2:
            k1k2k3k4.append(np.concatenate((kk,np.array([ii,jj]))))
        return k1k2k3k4, Vk1k2
    
    @njit
    def coulomb_matrix_elements(qlist, Vlist, Fmat_up, Fmat_dn):
        k1234_upup = List()
        Vc_upup = List()
        k1234_updn = List()
        Vc_updn = List()
        k1234_dnup = List()
        Vc_dnup = List()
        k1234_dndn = List()
        Vc_dndn = List()
        #temp_upup = List()
        for ii in range(n_k34):
            for jj in range(ii): # ii>jj; k3 > k4
                ##########################
                ### ii neq jj
                _,Vk1234_temp = create_table_k1234_from_k3k4(ii,jj,klist,qlist,Fmat_up, Fmat_up, Vlist)
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(jj,ii,klist,qlist,Fmat_up, Fmat_up, Vlist)
                k1234_upup.extend(k1k2k3k4)
                Vc_temp = List()
                for ll in range(len(Vk1234)): 
                    Vc_temp.append((Vk1234[ll] - Vk1234_temp[ll]))  
                Vc_upup.extend(Vc_temp) 
                ###########################
                _,Vk1234_temp = create_table_k1234_from_k3k4(ii,jj,klist,qlist,Fmat_dn, Fmat_dn, Vlist)
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(jj,ii,klist,qlist,Fmat_dn, Fmat_dn, Vlist)
                k1234_dndn.extend(k1k2k3k4)
                Vc_temp = List()
                for ll in range(len(Vk1234)): 
                    Vc_temp.append((Vk1234[ll] - Vk1234_temp[ll]))  
                Vc_dndn.extend(Vc_temp) 
        for ii in range(n_k34):
            for jj in range(n_k34):
                ###########################
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(ii,jj,klist,qlist,Fmat_up, Fmat_dn, Vlist, False)
                k1234_updn.extend(k1k2k3k4)
                Vc_updn.extend(Vk1234)      
                ###########################
                k1k2k3k4,Vk1234 = create_table_k1234_from_k3k4(ii,jj,klist,qlist,Fmat_dn, Fmat_up, Vlist, False)
                k1234_dnup.extend(k1k2k3k4)
                Vc_dnup.extend(Vk1234)

        return list(k1234_upup), list(Vc_upup), list(k1234_updn), list(Vc_updn), list(k1234_dnup), list(Vc_dnup), list(k1234_dndn), list(Vc_dndn)

    klist, qlist = sample_k_q(N1,N2,Nk)
    Vlist = coulomb_potential(qlist)
    Fmat_up, Fmat_dn, EEup, EEdn = compute_F_mat(klist)
    k1234_upup, Vc_upup, k1234_updn, Vc_updn, k1234_dnup, Vc_dnup, k1234_dndn, Vc_dndn = coulomb_matrix_elements(qlist,Vlist, Fmat_up, Fmat_dn)

    up_configs, config_arrays_up = bitstring_config(n_k34,Nup)
    dn_configs, config_arrays_dn = bitstring_config(n_k34,Ndn)
    spin_configs = configs_spin = [[configup, configdn] for configup in up_configs for configdn in dn_configs]
    Ktot_spin, configsGroupID_spin, singleE_spin = groupingKsum_spin(np.array(config_arrays_up), np.array(config_arrays_dn), klist, EEup, EEdn, N1, N2)
    solver = lambda gid: manybody_Hamiltonian_spinful(gid, spin_configs, singleE_spin, N1*N2, k1234_upup, Vc_upup, k1234_dndn, Vc_dndn, k1234_updn, Vc_updn, k1234_dnup, Vc_dnup)
    Evals_spin = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configsGroupID_spin)

    return  Ktot_spin, Evals_spin

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
    glist = np.array([[g1*N1,g2*N2] for g2 in glist for g1 in glist], dtype=np.int32)
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

def chernnum(NB,dim,HH, Q1, Q2, bandidx=None):
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
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])  #convert binary num to array

@njit
def groupingKsum(config_arrays, klist, EEup, N1, N2):
    ndim = config_arrays.shape[0]
    occ_ksum = np.zeros((ndim,2),dtype=np.int32)
    singleE = np.zeros(ndim, dtype=np.float64)
    EEup_flat = EEup.T.flatten()
    for jj in range(ndim):
        occ_ksum[jj] = np.sum(klist[config_arrays[jj]],axis=0)
        singleE[jj] = np.sum(EEup_flat[config_arrays[jj]])
    _,occ_ksum = compute_reduce_k(occ_ksum,N1,N2) 
    Ktot, grouped_k = two_index_unique(occ_ksum,N2)
    return Ktot,  grouped_k, singleE

@njit
def groupingKsum_spin(config_arrays_up, config_arrays_dn, klist, EEup, EEdn, N1, N2):
    ndim_up = config_arrays_up.shape[0]
    ndim_dn = config_arrays_dn.shape[0]
    ndim = ndim_up * ndim_dn
    occ_ksum = np.zeros((ndim,2),dtype=np.int32)
    singleE = np.zeros(ndim, dtype=np.float64)
    EEup_flat = EEup.T.flatten()
    EEdn_flat = EEdn.T.flatten()
    for jj in range(ndim_up):
        for ii in range(ndim_dn):
            occ_ksum[jj*ndim_dn + ii] = np.sum(klist[config_arrays_up[jj]], axis=0) + np.sum(klist[config_arrays_dn[ii]], axis=0)
            singleE[jj*ndim_dn + ii] = np.sum(EEup_flat[config_arrays_up[jj]], axis=0) + np.sum(EEdn_flat[config_arrays_dn[ii]], axis=0)
    _,occ_ksum = compute_reduce_k(occ_ksum,N1,N2) 
    Ktot, grouped_k = two_index_unique(occ_ksum,N2)
    return Ktot,  grouped_k, singleE


def manybody_Hamiltonian(configs_indx, up_configs, singleE, k1234, V1234):
    dimHam = configs_indx.shape[0]       
    configs =  itemgetter(*configs_indx)(up_configs)
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))
    Eigs = eigsh(matrix, k=4, which='SA', sigma=None, return_eigenvectors=False)
    return Eigs

def manybody_Hamiltonian_spinful(configs_indx, spin_configs, singleE, Nsys, k1234_upup, V1234_upup, k1234_dndn, V1234_dndn, k1234_updn, V1234_updn, k1234_dnup, V1234_dnup):
    dimHam = configs_indx.shape[0]       
    configs =  itemgetter(*configs_indx)(spin_configs)
    rows, cols, mat_ele = get_matrixEle_spinful(configs, Nsys, k1234_upup, V1234_upup, k1234_dndn, V1234_dndn, k1234_updn, V1234_updn, k1234_dnup, V1234_dnup)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))
    Eigs = eigsh(matrix, k=4, which='SA', sigma=None, return_eigenvectors=False)
    return np.sort(Eigs.real)

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
    plt.savefig("Energy_N=5x3_nup=3.pdf",dpi=300,bbox_inches='tight')
    return fig, ax

def plot_ED_energies(Ktot, Evals,N2):
    num_K = Ktot[:,0]*N2+Ktot[:,1]
    Evals_array = np.array(Evals)
    print("ktot,         Eigen values (meV)")
    for i in range(len(Ktot)):
        print(Ktot[i], Evals[i])
    minE = Evals_array.min()
    maxE = Evals_array.max()
    print()
    print("Eigenvalues sorted")
    print(np.sort(Evals_array.flatten()))
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(num_K, (Evals_array - minE),'k.',markersize=10)
    ax.set_ylabel(r'$E - E_{GS}(meV)$',fontsize=12)
    ax.set_xlabel(r'$k_1 N_2 + k_2$', fontsize=12)
    ax.set_ylim([-0.1,maxE-minE+0.2])
    return fig, ax

@njit
def find_id_in_array(inputconfig,configs):
    a = List()
    for config in inputconfig:
        a.append((configs == config).nonzero()[0][0])
    return a

theta = 2
Nx, Ny = 4, 4 
Nup, Ndn = 3, 2
Vd = 0
ti = time.time()
Ktot_spin, Evals_spin = main(theta,Nx,Ny,Nup,Ndn,Vd)
tf = time.time()
print("Total time:", tf-ti, "seconds.")
fig, ax = plot_ED_energies(Ktot_spin,Evals_spin,Ny)

#fig.savefig('EDspinful_N1='+str(Nx)+'_N2='+str(Ny)+'_Nup='+str(Nup)+'_Ndn='+str(Ndn)+'_ep=10_theta='+str(round(theta,2))+'_Vd='+str(round(Vd,2))+'.pdf',bbox_inches='tight',dpi=300)
plt.show()
