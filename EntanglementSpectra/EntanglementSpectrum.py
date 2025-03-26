from multiprocessing import Pool
import numpy as np
import time
from itertools import combinations,product
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.special import comb

sqrt_3=np.sqrt(3)
eigenvec1 = np.load('EigenVec1.npy')
eigenvec2 = np.load('EigenVec2.npy')
eigenvec3 = np.load('EigenVec3.npy')
basis1 = np.load('Basis.npy')
basis2 = np.load('Basis.npy')
basis3 = np.load('Basis.npy')
print('finish load1',flush=True)
Allowed_pairs=np.load('Allowed_pairs.npy',allow_pickle=True)
print('finish load2',flush=True)


#define the dictionary as a global variable
global basis1_to_eleid, basis2_to_eleid, basis3_to_eleid
basis1_to_eleid = {config: idx for idx, config in enumerate(basis1)}
del basis1
basis2_to_eleid = {config: idx for idx, config in enumerate(basis2)}
del basis2
basis3_to_eleid = {config: idx for idx, config in enumerate(basis3)}
del basis3

def bitstring_config(sys_size, num_particle):
    decimal_numbers = []
    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ['0'] * sys_size
        for idx in ones_indices:
            binary_str[idx] = '1'
        decimal_numbers.append(int(''.join(binary_str), 2))
    return decimal_numbers

def config_array(configs, Nsys):
    configs_bin = list(map(lambda x: format(x, '0' + str(Nsys) + 'b'), configs))
    # print("bin", configs_bin[0])
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])

def two_index_unique(configs,occ_array,Ni,direc_i):         #direc_i表示x,y方向，Ni表示i方向的格点数
    array_1d=flatten_to_1d(occ_array,Ni,direc_i)
    unique_1d=np.unique(array_1d)
    unique_configs = []
    for id in unique_1d:
        unique_indices=(array_1d == id).nonzero()[0]
        unique_configs.append(configs[unique_indices])
    return unique_configs

def groupingKsum(configs,mnklist,Nx,Ny):
    binary_lists=np.fliplr(config_array(configs,Nx*Ny))
    occ_ksum=np.dot(binary_lists,mnklist)
    _,K_fold=np.divmod(occ_ksum,np.array([Nx,Ny]))
    Kgroup=two_index_unique(configs,K_fold,Ny,1)
    return Kgroup

def flatten_to_1d(occ_array,Ni,direc_i):
    return occ_array[:,1-direc_i]*Ni+occ_array[:,direc_i]

def extend_to_2d(momentum_1d,Ni):
    x,y=np.divmod(momentum_1d,Ni)
    return np.vstack((x,y)).T


def sample_klist(Nx,Ny):
    k1list = np.arange(Nx, dtype=np.int32)  # *2*np.pi/N12
    k2list = np.arange(Ny, dtype=np.int32)  # *2*np.pi/N12
    k12list = np.array([np.array([k1list[ii], k2list[jj]]) for ii in range(Nx) for jj in range(Ny)],dtype=np.int32)
    return k12list



def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def get_bit_positions(n: int):
    return [i for i, bit in enumerate(bin(n)[:1:-1]) if bit == '1']

def getspin(b, i):  # spin (0 or 1) at 'i' in the basis 'b'
    return (b >> i) & 1

def exchange_coe(Aconfig,Bconfig):
    print(Aconfig&Bconfig)
    count_exchange=0
    stateB=Bconfig
    positions=get_bit_positions(stateB)
    positions1=get_bit_positions(Aconfig)
    for i in range(len(positions)):
        count_exchange+=bin(Aconfig & ((1 << positions[i]) - 1)).count('1')
        stateB &= stateB-1
    Sign=(-1)**count_exchange
    return Sign



def plot_ES(Ktot, Evals):
    num_K = Ktot.shape[0]
    Evals_array = Evals
    #print("ktot, ES")
    #for i in range(len(Ktot)):
       #print(Ktot[i], Evals[i])
    fig, ax = plt.subplots()
    for i, evals in enumerate(Evals_array):
        index = Ktot[i][0] + Nx * Ktot[i][1]
        #print(index,evals)
        ax.plot([index] * len(evals), evals, 'r_')
    ax.set_ylabel(r'$\Xi$')
    ax.set_xlabel(r'$k_1 + k_2*N_1$')
    return fig, ax

if __name__== '__main__':
    Nx= 3
    Ny = 6
    Ns0 = Nx * Ny

    Nparticles = 6
    NA = 3
    NB = Nparticles - NA
    NomalizeCoe=np.sqrt(comb(Nparticles, NA, exact=True))

    klist = sample_klist(Nx, Ny)
    configsA = np.array(bitstring_config(Ns0, NA))
    KgroupsA = groupingKsum(configsA, klist, Nx, Ny)
    if NA==NB:
        configsB = np.concatenate(KgroupsA)
    else:
        configsB = np.array(bitstring_config(Ns0, NB))

    RDM_eigen=[]
    Eigen=[]
    for gid in range(Ns0):
        print('gid:', gid, flush=True)
        row_id = []
        col_id = []
        element = []
        Aconfig = KgroupsA[gid]
        pairs_indexes = np.load(f'Marix_index{gid}.npy')
        pairs_configs = Allowed_pairs[gid]
        num_p = len(pairs_configs)
        rownumber = len(Aconfig)
        colnumber = len(configsB)


        def process_chunk(start, end):
            chunk_element = []
            chunk_row_id = []
            chunk_col_id = []
            for id in range(start, end):
                config = pairs_configs[id]
                if config in basis1_to_eleid:
                    ele_id = basis1_to_eleid[config]
                    row_id = pairs_indexes[0, id]
                    col_id = pairs_indexes[1, id]
                    coefficient = exchange_coe(Aconfig[row_id], configsB[col_id])
                    chunk_element.append(eigenvec1[ele_id] / (NomalizeCoe * sqrt_3)*coefficient)
                    chunk_row_id.append(row_id)
                    chunk_col_id.append(col_id)

                if config in basis2_to_eleid:
                    ele_id = basis2_to_eleid[config]
                    row_id = pairs_indexes[0, id]
                    col_id = pairs_indexes[1, id]
                    coefficient = exchange_coe(Aconfig[row_id], configsB[col_id])
                    chunk_element.append(eigenvec2[ele_id] / (NomalizeCoe * sqrt_3) * coefficient)
                    chunk_row_id.append(row_id)
                    chunk_col_id.append(col_id)

                if config in basis3_to_eleid:
                    ele_id = basis3_to_eleid[config]
                    row_id = pairs_indexes[0, id]
                    col_id = pairs_indexes[1, id]
                    coefficient = exchange_coe(Aconfig[row_id], configsB[col_id])
                    chunk_element.append(eigenvec3[ele_id] / (NomalizeCoe * sqrt_3) * coefficient)
                    chunk_row_id.append(row_id)
                    chunk_col_id.append(col_id)

            return chunk_element, chunk_row_id, chunk_col_id

        # 分块处理
        chunk_size = num_p // 64 # 假设分成8块
        with Pool(64) as pool:
            results = pool.starmap(process_chunk, [(i, min(i + chunk_size, num_p)) for i in range(0, num_p, chunk_size)])
        print('finish',gid)
        # 合并结果
        element = np.concatenate([result[0] for result in results])
        row_id = np.concatenate([result[1] for result in results])
        col_id = np.concatenate([result[2] for result in results])
        DMatrix = sp.coo_matrix((element, (row_id, col_id)), shape=(rownumber, colnumber))
        DM_dag = DMatrix.conj().T
        RDMatrix = np.dot(DMatrix, DM_dag)
        RDMatrix = RDMatrix.toarray()
        eigenvalues = np.linalg.eigvals(RDMatrix)
        epsilon1 = -np.log(np.real(eigenvalues))
        RDM_eigen.append(epsilon1)
        Eigen.append(np.sum(np.real(eigenvalues)))
        print('sum e^i\zeta', np.sum(np.real(eigenvalues)))

    #Entropy=np.array([results[ii][1] for ii in range(Ns0)])
    print('SumEigen', np.sum(Eigen),flush=True)
    np.save('Spectrum1.npy', np.array(RDM_eigen,dtype=object))
    fig, ax = plot_ES(klist, RDM_eigen)
    picturename = f"Checkboard_ES.png"
    plt.savefig(picturename)
    plt.show()
