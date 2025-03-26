import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy import sin, cos, exp, pi, sqrt
from joblib import Parallel, delayed
import multiprocessing
from itertools import combinations
from numba import jit, njit
from numba.typed import List
#from pylanczos import PyLanczos
import time
import sys

num_cores = multiprocessing.cpu_count()

def sample_klist(Nx,Ny):
    k1list = np.arange(Nx, dtype=np.int32)  # *2*np.pi/N12
    k2list = np.arange(Ny, dtype=np.int32)  # *2*np.pi/N12
    k12list = np.array([np.array([k1list[ii], k2list[jj]]) for ii in range(Nx) for jj in range(Ny)],dtype=np.int32)
    return k12list

def main(Nx, Ny,Nup):

    n_k34=Nx*Ny
    FactorU = 0.5 / n_k34
    Nk=1
    n_g=(2*Nk+1)**2

    Tranarray=np.array([[Nx,0],[0,Ny]])


    def GetLinearizedMomentum(Kx,Ky):
        return (Ky*Nx+Kx)

    def ComputeProjectedMomenta(dkx,dky):
        ProjectedMomenta=np.zeros((n_k34,2),dtype=np.float64)
        Nx1=Nx
        Ny1=0
        Nx2=0
        Ny2=Ny
        for Kx in range(Nx):
            for Ky in range(Ny):
                kx_trans=Kx+dkx#+Offset    #晶格扭曲
                ky_trans=Ky+dky
                projectedMomentum0 = 2.0 * np.pi*(kx_trans*Ny2-ky_trans*Ny1)/(Nx*Ny)
                projectedMomentum1 = 2.0 * np.pi*(-kx_trans*Nx2+ky_trans*Nx1)/(Nx*Ny)
                ProjectedMomenta[GetLinearizedMomentum(Kx,Ky)][0]=projectedMomentum0
                ProjectedMomenta[GetLinearizedMomentum(Kx, Ky)][1] = projectedMomentum1
        return ProjectedMomenta

    def GetProjectedMomentum(Kx,Ky,Latticecomponent):
        dkx=0
        dky=0
        ProjectIndex=GetLinearizedMomentum(Kx,Ky)
        ProjectedMomenta = ComputeProjectedMomenta(dkx,dky)
        return ProjectedMomenta[int(ProjectIndex)][Latticecomponent]

    def ComputeTwoBodyMatrixElementAB(kx1, ky1, kx2, ky2):
        kpx1=GetProjectedMomentum(kx1,ky1,0)
        kpy1=GetProjectedMomentum(kx1,ky1,1)
        kpx2=GetProjectedMomentum(kx2,ky2,0)
        kpy2=GetProjectedMomentum(kx2,ky2,1)
        Tmp=2.0*(np.cos(0.5*(kpx2-kpy2+kpy1-kpx1))+np.cos(0.5*(kpx2+kpy2-kpy1-kpx1)))
        #print(k_x1,k_y1,k_x2,k_y2,Tmp)
        return Tmp

    def Compute_F_mat():
        OnebodyBasis = np.zeros((Nx*Ny, 2), dtype=complex)
        OnebodyBasis[0] = [-1j * 0.70710678118655, 1j * 0.70710678118655]
        OnebodyBasis[1] = [-1j * 0.6638044134011, 1j * 0.74790621119845]
        OnebodyBasis[2] = [-1j * 0.48592719226106, 1j * 0.87399929280365]
        OnebodyBasis[3] = [0, -1j]
        OnebodyBasis[4] = [0.48592719226106, 0.87399929280365]
        OnebodyBasis[5] = [1j * 0.6638044134011, 1j * 0.74790621119845]
        OnebodyBasis[6] = [-1j * 0.87399929280365, 1j * 0.48592719226106]
        OnebodyBasis[7] = [0.52277365269719 - 1j * 0.62200991033436, -0.58075549716409 + 1j * 0.05034314258336]
        OnebodyBasis[8] = [0.47140950751328 - 1j * 0.5270418163923, -0.6490685440256 - 1j * 0.28055307012487]
        OnebodyBasis[9] = [1j * 0.6638044134011, 0.74790621119845]
        OnebodyBasis[10] = [-1j * 0.70710678118655, -0.6708203932499 - 1j * 0.22360679774998]
        OnebodyBasis[11] = [0.64935112189432 - 1j * 0.48839711404052, 0.081652923635761 - 1j * 0.57718643396498]
        OnebodyBasis[12] = [-1j * 0.87399929280365, -1j * 0.48592719226106]
        OnebodyBasis[13] = [-0.81251992006875 + 1j * 8.4320616582892e-17, -0.41219617871317 + 1j * 0.41219617871317]
        OnebodyBasis[14] = [0.68599434057004 - 1j * 0.17149858514251, 0.054232614454664 - 1j * 0.70502398791063]
        OnebodyBasis[15] = [-1j * 0.6638044134011, -0.74790621119845]
        OnebodyBasis[16] = [1j * 0.70710678118655, 0.67082039324994 - 1j * 0.22360679774998]
        OnebodyBasis[17] = [0.085094043522528 + 1j * 0.80805174603209, 0.36676073408819 - 1j * 0.45309816091281]

        VV=OnebodyBasis.reshape([Nx,Ny,2],order='F')
        Fmat = np.zeros([Nx, Ny,Nx,Ny,2], dtype=np.complex128)
        Fmat[:,:,:,:,:]=np.einsum('abm,cdm->abcdm',VV[:,:,:].conj(),VV[:,:,:])
        Fmatflat=Fmat.flatten(order='F')

        return Fmatflat

    def Coulomb_potential(k1id,k2id,klist):
        Vc=[]
        # k1=klist[k1id]
        # k2=klist[k2id]
        # vc = ComputeTwoBodyMatrixElementAB(k1[0], k1[1], k2[0], k2[1])
        # Vc.append(vc)
        for id in k1id:
            k1=klist[id]
            k2=klist[k2id]
            vc=ComputeTwoBodyMatrixElementAB(k1[0],k1[1],k2[0],k2[1])
            Vc.append(vc)
        return np.array(Vc)

    def find_id_in_klist(braket_k):
        kid = np.array([klist_lookup[tuple(bk)] for bk in braket_k])
        return kid

    def find_id_in_glist(g_k):
        g_k += 1
        return g_k[:, 0] + (2*Nk+1) * g_k[:, 1]

    def get_k_q_ids(k3_id, k4_id, klist, qlist):
        Tranarray=np.array([[Nx,0],[0,Ny]])
        k3_q = klist[k3_id] + qlist
        k4_q = klist[k4_id] - qlist
        g_k3_q, braket_k3_q = compute_folded_k(k3_q,klist,Tranarray)
        g_k4_q, braket_k4_q = compute_folded_k(k4_q,klist,Tranarray)
        k3_q_id = find_id_in_klist(braket_k3_q)
        k4_q_id = find_id_in_klist(braket_k4_q)
        g_k3_q_id = find_id_in_glist(g_k3_q)
        g_k4_q_id = find_id_in_glist(g_k4_q)
        return k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id

    def compute_coulomb(k3_id, k4_id, klist,qlist, Fmat1, Fmat2):
        k3_q_id, g_k3_q_id, k4_q_id, g_k4_q_id = get_k_q_ids(k3_id, k4_id, klist, qlist)  #这里的k3/4_q_id实际上是k1/2_q_id
        ind_k = (g_k3_q_id <n_g) & (g_k3_q_id >= 0) & (g_k4_q_id <n_g) & (g_k4_q_id >= 0) & (k3_q_id < k4_q_id)
        k3_q_id = k3_q_id[ind_k]
        k4_q_id = k4_q_id[ind_k]
        if len(k3_q_id)>0 and len(k4_q_id)>0:
            # k3_q_id=1
            # k4_q_id=5
            # k3_id=1
            # k4_id=5
            Vlist1 = Coulomb_potential(k4_q_id, k4_id, klist)
            V1 = FactorU*Fmat1[ k3_id * n_k34 + k3_q_id] * Fmat2[n_k34**2+ k4_id * n_k34 + k4_q_id]*Vlist1
            Vlist2 = Coulomb_potential(k3_q_id, k4_id, klist)
            V2 = FactorU*Fmat1[k3_id * n_k34 + k4_q_id] * Fmat2[n_k34 ** 2 + k4_id * n_k34 + k3_q_id] * Vlist2
            Vlist3 = Coulomb_potential(k4_q_id, k3_id, klist)
            V3 = FactorU * Fmat1[k4_id * n_k34 + k3_q_id] * Fmat2[n_k34 ** 2 + k3_id * n_k34 + k4_q_id] * Vlist3
            Vlist4 = Coulomb_potential(k3_q_id, k3_id, klist)
            V4 = FactorU * Fmat1[k4_id * n_k34 + k4_q_id] * Fmat2[n_k34 ** 2 + k3_id * n_k34 + k3_q_id] * Vlist4
            V=V1-V2-V3+V4
            k1k2list = np.vstack((k3_q_id, k4_q_id)).T
            k1k2, k1k2_ids = two_index_unique(k1k2list, n_k34)
            Vk1k2 = List()
            for k_id in k1k2_ids:
                Vk1k2.append(np.sum(V[k_id]))
        else:
            return None,np.array([100, 100])
        return Vk1k2, k1k2

    def create_table_k1234_from_k3k4(ii, jj, klist, qlist, Fmat_1, Fmat_2):
        k1k2k3k4 = List()
        Vk1k2, k1k2 = compute_coulomb(ii, jj, klist,qlist, Fmat_1, Fmat_2)
        if Vk1k2==None:
            return k1k2,Vk1k2
        for kk in k1k2:
            k1k2k3k4.append(np.concatenate((kk, np.array([ii, jj]))))
        return k1k2k3k4, Vk1k2

    def coulomb_matrix_elements(klist,qlist,  Fmat_up):
        k1234_upup = List()
        Vc_upup = List()
        for ii in range(n_k34):
            for jj in range(ii):
                ##########################
                ### ii neq jj
                #k1234_temp, Vk1234_temp = create_table_k1234_from_k3k4(ii, jj, klist, qlist, Fmat_up, Fmat_up)
                # if Vk1234_temp == None:
                #     continue
                k1k2k3k4, Vk1234 = create_table_k1234_from_k3k4(jj, ii, klist, qlist, Fmat_up, Fmat_up)
                if Vk1234 == None:
                    continue
                k1234_upup.append(k1k2k3k4)
                # Vc_upup.append(Vk1234)
                Vc_temp = List()
                for ll in range(len(Vk1234)):
                    Vc_temp.append( (Vk1234[ll]*2))#- Vk1234_temp[ll]
                Vc_upup.append(Vc_temp)
        return k1234_upup, Vc_upup

    def create_coulomb_table(k1234_upup, Vc_upup):
        return np.vstack(k1234_upup), np.hstack(Vc_upup)

    klist = sample_klist(Nx, Ny)
    qlist = klist
    klist_lookup = {tuple(value): index for index, value in enumerate(klist)}

    up_configs = np.array(bitstring_config(n_k34, Nup))
    Ktot, configsGroupID = groupingKsum(up_configs, klist, Nx, Ny, Tranarray)

    Fmat_up=Compute_F_mat()

    k1234_upup, Vc_upup = coulomb_matrix_elements(klist, qlist, Fmat_up)
    k1234_upup, Vc_upup = create_coulomb_table(k1234_upup, Vc_upup)


    Evals=manybody_Hamiltonian(configsGroupID[3],up_configs,k1234_upup,np.array(Vc_upup, dtype=np.complex128))
    # solver = lambda gid: manybody_Hamiltonian(gid, up_configs,k1234_upup,
    #                                           np.array(Vc_upup, dtype=np.complex128))
    # Evals = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configsGroupID)

    return Ktot,Evals

def manybody_Hamiltonian(configs_indx, up_configs, k1234, V1234):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend([0]*dimHam)
    matrix = sp.csc_matrix((mat_ele, (rows, cols)))
    Eigs, Vec = eigsh(matrix, k=5, which='SA', sigma=None, return_eigenvectors=True)
    idx = Eigs.argsort()
    Eigs = Eigs[idx]
    Eigvecs = Vec[:, idx]
    folder_path = f'./'
    filename1 = f'{folder_path}EigenVec1.npy'
    filename2 = f'{folder_path}EigenVec2.npy'
    filename3 = f'{folder_path}EigenVec3.npy'
    filename4 = f'{folder_path}Basis.npy'
    np.save(filename1, Eigvecs[:, 0])
    np.save(filename2, Eigvecs[:, 1])
    np.save(filename3, Eigvecs[:, 2])
    np.save(filename4, configs)
    return Eigs

def getspin(b, i):  # spin (0 or 1) at 'i' in the basis 'b'
    return (b >> i) & 1

def bitflip(b, i):  # Flips bits (1-->0, 0-->1) at loc 'i'
    return b ^ (1 << i)

def spinless_fsigns(config, indices):  # for spinless config Ck1dag Ck2dag Ck4 Ck3
    i = min(indices[0], indices[1])
    j = max(indices[0], indices[1])
    k = min([indices[2], indices[3]])
    l = max([indices[2], indices[3]])
    Fkl = np.array([getspin(config, x) for x in range(k + 1, l)])
    Fkl = np.prod(1 - 2 * Fkl)
    new_config = bitflip(bitflip(config, k), l)
    Fij = np.array([getspin(new_config, y) for y in range(i + 1, j)])
    Fij = np.prod(1 - 2 * Fij)
    new_config = bitflip(bitflip(new_config, i), j)
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

def get_matrixEle(configs, k1234, V1234):
    config_lookup = {value: index for index, value in enumerate(configs)}
    Matele = []
    row = []
    col = []
    for ks, V in zip(k1234, V1234):
        configs_k3 = getspin(configs, ks[2])
        configs_k4 = getspin(configs, ks[3])
        valid_k34_id = ((configs_k3 * configs_k4) == 1).nonzero()[0]

        configs_k1 = getspin(configs[valid_k34_id], ks[0])
        configs_k2 = getspin(configs[valid_k34_id], ks[1])
        valid_kid1 = ((configs_k1 + configs_k2) == 0).nonzero()[0]

        valid_kid2 = (((ks[0] == ks[2]) or (ks[0] == ks[3])) & (configs_k2 == 0)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k34_id[valid_kid1], valid_k34_id[valid_kid2]))

        valid_kid3 = (((ks[1] == ks[2]) or (ks[1] == ks[3])) & (configs_k1 == 0)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid3]))

        valid_kid4 = ((ks[0] == ks[2]) & (ks[1] == ks[3]) & (configs_k2 == 1)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid4]))

        valid_kid5 = ((ks[0] == ks[3]) & (ks[1] == ks[2]) & (configs_k2 == 1)).nonzero()[0]
        valid_k_id = np.concatenate((valid_k_id, valid_k34_id[valid_kid5]))

        for jj in valid_k_id:
            config = configs[jj]
            fsign, newconfig = spinless_fsigns(config, ks)
            Matele.append(fsign * V)
            col.append(jj)
            row.append(config_lookup[newconfig])
    return row, col, Matele


def config_array(configs, Nsys):
    configs_bin = list(map(lambda x: format(x, '0' + str(Nsys) + 'b'), configs))
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])

def groupingKsum(configs, mnklist, N1, N2,Transarray):
    binary_lists = np.fliplr(config_array(configs, N1 * N2))  # convert binary num to array
    occ_ksum = binary_lists @ mnklist
    _, occ_ksum = compute_folded_k(occ_ksum,mnklist,Transarray)
    #negtive_x = occ_ksum[occ_ksum[:, 0] < 0]
    #negtive_y = occ_ksum[occ_ksum[:, 1] < 0]
    #minneg_x = min(negtive_x[:, 0])
    #minneg_y = min(negtive_y[:, 1])
    #occ_ksum += np.array([-minneg_x, -minneg_y])
    Ktot, grouped_k = two_index_unique(occ_ksum, N2 * N1)
    #Ktot -= np.array([-minneg_x, -minneg_y])
    #E = EEup.flatten(order='F')
    #singleE = binary_lists @ EEup.flatten(order='F')
    return Ktot, grouped_k

def bitstring_config(sys_size, num_particle):
    decimal_numbers = []
    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ['0'] * sys_size
        for idx in ones_indices:
            binary_str[idx] = '1'
        decimal_numbers.append(int(''.join(binary_str), 2))
    return decimal_numbers

def convert_to_1_id(k1k2list, N):
    return k1k2list[:, 0] * N + k1k2list[:, 1]

def convert_to_2_id(klist, N):
    x, y = np.divmod(klist, N)
    return np.vstack((x, y)).T

def two_index_unique(testarray, N):
    array_1d = convert_to_1_id(testarray, N)
    unique_id = np.unique(array_1d)
    unique_indices = List()
    for uid in unique_id:
        unique_indices.append((array_1d == uid).nonzero()[0])
    unique_2d = convert_to_2_id(unique_id, N)
    return unique_2d, unique_indices

def compute_folded_k(kqlist,klist,Transarray):
    g1 = Transarray[0]
    g2 = Transarray[1]
    kqlist = np.array(kqlist, dtype=np.float64) #+ 1e-5
    g_k_q=np.zeros((len(kqlist),2),dtype=float)
    braket = np.zeros((len(kqlist), 2), dtype=int)
    # 构造矩阵A和向量P
    A = np.array([g1, g2], dtype=np.float64)
    A_inv = np.linalg.inv(A)
    for k in klist:
        comparelist=kqlist-k
        coefficients = np.dot(comparelist, A_inv)
        integer_row = np.all(np.isclose(coefficients, np.round(coefficients)),axis=1)
        integer_indices=np.where(integer_row)[0]
        a_fold=coefficients[integer_row][:,0]
        b_fold =coefficients[integer_row][:,1]
        g_k_q[integer_indices]=np.column_stack((a_fold,b_fold))
        braket[integer_indices]=k
    g_k_q=g_k_q.round().astype(np.int64)
    return g_k_q, braket

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
    # print(Evals_array - minE)
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_K), (Evals_array - minE), 'r.', markersize=10)
    ax.set_ylabel(r'$E - E_{GS}(meV)$')
    ax.set_xlabel(r'$k_1 N_2 + k_2$')
    ax.set_ylim([0, 0.2])
    return fig, ax

Nx=3
Ny=6
Nup=6
Ktot,Evals=main(Nx,Ny,Nup)
#fig, ax = plot_ED_energies(Ktot, Evals)
#picturename = f"Checkboard_Spectrum.png"
#plt.savefig(picturename)
