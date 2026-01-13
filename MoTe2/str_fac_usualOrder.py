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

num_cores = multiprocessing.cpu_count()

epsilon_r = 10.0 ##float(sys.argv[1])  #coulomb parameter
def main(theta, N1, N2, Nup, Vd):
    theta = theta/180.0*pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1,Q2 = reciprocal_vecs(gvecs)
    aM = a0/(2*np.sin(theta/2))
    print('aM', aM)
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
        return ham_x, ham_y

    def mat_ele_H_der(k, psi0, psi1):
        ham_x, ham_y = H_der(k)
        return psi0.conj()@ham_x@psi1, psi0.conj()@ham_y@psi1
    
    def compute_F_mat(klist):
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
    
        VV = np.array([result[jj][1] for jj in range(n_k34)], dtype=np.complex128).reshape([N1,N2,2*Nk+1, 2*Nk+1, 2, dim],order='F')[:,:,:,:,:,-1]
        Fmat = np.zeros([N1,N2,N1,N2,2*Nk+1,2*Nk+1], dtype=np.complex128)
        Fmat[:,:,:,:,Nk,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:,:].conj(), VV[:,:,:,:,:])
        # for jj in range(1,Nk+1):
        #     for ii in range(1,Nk+1):
        #         Fmat[:,:,:,:,Nk-ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,jj:,:].conj(), VV[:,:,:(-ii),:(-jj),:])
        #         Fmat[:,:,:,:,Nk-ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,ii:,:(-jj):,:].conj(), VV[:,:,:(-ii),jj:,:])
        #         Fmat[:,:,:,:,Nk+ii,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),jj:,:].conj(), VV[:,:,ii:,:(-jj),:])
        #         Fmat[:,:,:,:,Nk+ii,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-ii),:(-jj),:].conj(), VV[:,:,ii:,jj:,:])
        # for jj in range(1,Nk+1):
        #     Fmat[:,:,:,:,Nk-jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,jj:,:,:].conj(), VV[:,:,:(-jj),:,:])
        #     Fmat[:,:,:,:,Nk+jj,Nk] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:(-jj),:,:].conj(), VV[:,:,jj:,:,:])
        #     Fmat[:,:,:,:,Nk,Nk-jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,jj:,:].conj(), VV[:,:,:,:(-jj),:])
        #     Fmat[:,:,:,:,Nk,Nk+jj] = np.einsum('abmnl,cdmnl->abcd', VV[:,:,:,:(-jj),:].conj(), VV[:,:,:,jj:,:])
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

        #print("qvals= ",absqlist[3300])
        
        k0 = 8.99e9
        J_to_meV = 6.242e21  #meV
        e_charge = 1.602e-19 # coulomb
        #epsilon = 8.854e-12
        #epsilon_r = 5
        Area = np.sqrt(3)/2*N1*N2*aM**2
        Vc = 2*np.pi*e_charge**2/(epsilon_r*absqlist)*J_to_meV/Area*k0  
    
        #Vc = 6.84877*1e-2/absqlist 
        #print("constant", Vc[:100])
        #In CGS unit
        #e =  4.8032046e-10 #statcoulomb
        #erg_to_meV = 6.241506e+14
        #Area = np.sqrt(3)/2*N1*N2*aM**2*100 
        #epsilon_r = 10
        #Vc = 2*np.pi*e*e/(Area*epsilon_r*absqlist)*erg_to_meV
        #"""
        Vc[zeroidx]=0
    
        return Vc
        

    def bandstructure(energy_range=None):
        npts = 40
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
    
    def JJ_single_mat_ele(Uvecs, klist):
        return "to be constructed"

    def str_fac(qval, Evec, Uvecs, config, klist, Len):
        config_lookup = {value: index for index, value in enumerate(config)}
        k_ind = find_id_in_klist(klist.copy())
        k2_minus_q = klist - qval
        gk_m_q,k2_minus_q_mBZ = compute_reduce_k(k2_minus_q, N1, N2)
        k2_minus_q_id = find_id_in_klist(k2_minus_q_mBZ)
        gk2_m_q_id = find_id_in_glist(gk_m_q.copy())

        k1_plus_q = klist + qval
        gk_p_q,k1_plus_q_mBZ = compute_reduce_k(k1_plus_q, N1, N2)
        k1_plus_q_id = find_id_in_klist(k1_plus_q_mBZ)
        gk1_p_q_id = find_id_in_glist(gk_p_q.copy())

        ind_k2 = (gk2_m_q_id<n_g) & (gk2_m_q_id>=0)
        ind_k1 = (gk1_p_q_id<n_g) & (gk1_p_q_id>=0)

        k2_minus_q_id = k2_minus_q_id[ind_k2]
        gk2_m_q_id = gk2_m_q_id[ind_k2]
        k1_plus_q_id = k1_plus_q_id[ind_k1]
        gk1_p_q_id = gk1_p_q_id[ind_k1]

        k1_ind = k_ind[ind_k1]
        k2_ind = k_ind[ind_k2]

        kxylist = klist.copy().astype(float)
        kxylist[:,0] = klist[:,0]/N1
        kxylist[:,1] = klist[:,1]/N2
        kxylist = kxylist @ np.array([Q1,Q2])
        
        origin_id = int((n_g-1)/2) #this is location of g=[0,0] in glist

        Sq = 0.0
        for i, k2 in enumerate(k2_ind):
            shift = origin_id - gk2_m_q_id[i]  #shift g_k --> g_k + g_(k+q)
            uvec = Uvecs[k2_minus_q_id[i]].copy()
            uvec[:n_g] = np.roll(uvec[:n_g],shift)
            uvec[n_g:] = np.roll(uvec[n_g:],shift)
            lamda1 =  uvec.conj() @ Uvecs[k2]
           
            ## order of mb operator cdag(k1+q)c(k1)cdag(k2-q)c(k2)
            config_k2 = getspin(config,k2)
            valid_config_k2ID = (config_k2==1).nonzero()[0]
            f1, new_config = c_an(config[valid_config_k2ID], k2)
            valid_config_k2mqID = (getspin(new_config, k2_minus_q_id[i])==0).nonzero()[0]
            f2, new_config = c_dag(new_config[valid_config_k2mqID], k2_minus_q_id[i])
            f2 = f1[valid_config_k2mqID] * f2

            for j, k1 in enumerate(k1_ind):
                valid_config_k1ID = (getspin(new_config, k1)==1).nonzero()[0]
                f3, newconfig = c_an(new_config[valid_config_k1ID], k1)
                f3 = f2[valid_config_k1ID] * f3
                valid_config_k1pqID = (getspin(newconfig, k1_plus_q_id[j])==0).nonzero()[0]
                f4, newconfig = c_dag(newconfig[valid_config_k1pqID], k1_plus_q_id[j])
                fsign = f3[valid_config_k1pqID] * f4
                
                #the components of evec that gives non-zero expectation values
                original_idlist = valid_config_k2ID[valid_config_k2mqID[valid_config_k1ID[valid_config_k1pqID]]]
                newconfig_id = np.array([config_lookup[nc] for nc in newconfig]).astype(int)

                u_vec = Uvecs[k1_plus_q_id[j]].copy()
                Shift = origin_id - gk1_p_q_id[j]
                u_vec[:n_g] = np.roll(u_vec[:n_g], Shift)
                u_vec[n_g:] = np.roll(u_vec[n_g:], Shift)
                lamda2 =  u_vec.conj() @ Uvecs[k1]

                if Len==1:
                    for l in range(len(Evec[0])):
                        mb_term =  sum(Evec[:,l][newconfig_id].flatten().conjugate() * Evec[:,l][original_idlist].flatten() * fsign)
                        Sq += lamda1 * lamda2 * mb_term
                else:
                    mb_term = sum(Evec[newconfig_id].flatten().conjugate() * Evec[original_idlist].flatten() * fsign)
                    Sq += lamda1 * lamda2 * mb_term

        return Sq

    def compute_str_fac(qval, Evecs, Uvecs, configs, klist):
        Len = len(configs)
        JJ_corr = 0.0
        if Len ==1:
            JJ_corr = str_fac(qval, Evecs, Uvecs, configs[0], klist, Len)
            JJ_corr = JJ_corr.real/len(Evecs[0])  #averaging over degenerate ground states
        else:
            for p in range(Len):
                JJ_corr += str_fac(qval, Evecs[p], Uvecs, configs[p], klist, Len)
            JJ_corr = JJ_corr.real/Len
        return JJ_corr
    
    klist, qlist = sample_k_q(N1,N2,Nk)
    Vlist = coulomb_potential(qlist)
    Fmat_up, EEup, Uvecs = compute_F_mat(klist)
    k1234_upup, Vc_upup = coulomb_matrix_elements(qlist, Vlist, Fmat_up)
    k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)
    
    up_configs = np.array(bitstring_config(n_k34,Nup))
    Ktot, configsGroupID, singleE = groupingKsum(up_configs, klist, EEup, N1, N2)
    Evals = []

    Gs_Ktot_ind = [0,0,0] #degenerate GS k indices
    configGid = []
    configurs = []
    for g in Gs_Ktot_ind:
        configid = configsGroupID[g]
        configGid.append(configid)
        configurs.append(up_configs[configid])

    ## Diagonalizing Hamiltonian
    if len(Gs_Ktot_ind)>1: # If degeneracy happens at diff k points
        solver = lambda gid: mbHamiltonian(len(Gs_Ktot_ind),gid, up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))
        Evecs = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configGid)
    else: # If degeneracy happens at same k point
        Evecs = mbHamiltonian(len(Gs_Ktot_ind),configGid[0], up_configs, singleE, k1234_upup, np.array(Vc_upup, dtype=np.complex128))

    solver = lambda q: compute_str_fac(q, Evecs, Uvecs, configurs, klist)
    Sq =  np.array(Parallel(n_jobs=num_cores)(delayed(solver)(q) for q in klist))
    Sq[0] -= Nup**2
    Sq = Sq/(N1*N2)
    for i in range(len(klist)):
        print(klist[i], Sq[i])
    return klist, Sq


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

def groupingKsum(configs, klist, EEup, N1, N2):
    dimH = len(configs)
    binary_lists = np.fliplr(config_array(configs,N1*N2)) #convert binary num to array
    occ_ksum =binary_lists@klist
    _,occ_ksum = compute_reduce_k(np.array(occ_ksum),N1,N2) 
    Ktot, grouped_k = two_index_unique(occ_ksum,N2)
    singleE = binary_lists @ EEup.flatten(order='F')
    #print(singleE)
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


def mbHamiltonian(Len,configs_indx, up_configs, singleE, k1234, V1234):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))

    if Len > 1: 
        Eigs,vecs = eigsh(matrix,k=1,which='SA',sigma=None,return_eigenvectors=True)
    else:
        Eigs,vecs = eigsh(matrix,k=3,which='SA',sigma=None,return_eigenvectors=True)
    #Eigs, vecs = np.linalg.eigh(matrix.toarray())
    print(Eigs.real)
    return vecs


def manybody_Hamiltonian(configs_indx, up_configs, singleE, k1234, V1234):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))
    
    # shape = (dimHam,dimHam)
    Eigs = eigsh(matrix,k=5,which='SA',sigma=None,return_eigenvectors=False)
    Eigs = np.sort(Eigs.real)
    #engine = PyLanczos(matrix, False, 6)
    #Eigs,_ = engine.run()
    #Eigs = Eigs.real
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
    ax.plot(np.arange(num_K), (Evals_array - minE),'k.',markersize=10)
    ax.set_ylabel(r'$E - E_{GS}(meV)$')
    ax.set_xlabel(r'$k_1 N_2 + k_2$')
    ax.set_ylim([-0.1,maxE-minE+0.2])
    return fig, ax

theta = 2.0 ##float(sys.argv[2])
Nx, Ny, Nhole = 3, 3, 3
Vd = 0.0 ##float(sys.argv[3]) #displacement field
k_list,Sq =  main(theta,Nx, Ny, Nhole, Vd)
plt.show()
