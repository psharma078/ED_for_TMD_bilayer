import numpy as np
from itertools import combinations,product
from joblib import Parallel, delayed

def bitstring_config(sys_size, num_particle):
    decimal_numbers = []
    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ['0'] * sys_size
        for idx in ones_indices:
            binary_str[idx] = '1'
        decimal_numbers.append(int(''.join(binary_str), 2))
    return decimal_numbers

def sample_klist(Nx,Ny):
    k1list = np.arange(Nx, dtype=np.int32)
    k2list = np.arange(Ny, dtype=np.int32)
    k12list = np.array([np.array([k1list[ii], k2list[jj]]) for ii in range(Nx) for jj in range(Ny)],dtype=np.int32)
    return k12list

def config_array(configs, Nsys):
    configs_bin = list(map(lambda x: format(x, '0' + str(Nsys) + 'b'), configs))
    # print("bin", configs_bin[0])
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])

def flatten_to_1d(occ_array,Ni,direc_i):
    return occ_array[:,1-direc_i]*Ni+occ_array[:,direc_i]

def two_index_unique(configs,occ_array,Ni,direc_i):         #direc_i表示x,y方向，Ni表示i方向的格点数
    array_1d=flatten_to_1d(occ_array,Ni,direc_i)
    unique_1d=np.unique(array_1d)
    #unique_indices = []
    unique_configs = []
    for id in unique_1d:
        #unique_indices.append((array_1d == id).nonzero()[0])
        unique_indices=(array_1d == id).nonzero()[0]
        unique_configs.append(configs[unique_indices])
    #unique_2d=extend_to_2d(unique_1d,Ni)
    return unique_configs

def groupingKsum(configs,mnklist,Nx,Ny):
    binary_lists=np.fliplr(config_array(configs,Nx*Ny))
    occ_ksum=np.dot(binary_lists,mnklist)
    _,K_fold=np.divmod(occ_ksum,np.array([Nx,Ny]))
    Kgroup=two_index_unique(configs,K_fold,Ny,1)
    return Kgroup

def Repeated_configs(Aconfig,KgroupsB):
    Allowed_configs=[]
    pair_index=[]
    count=0
    for Bconfig in KgroupsB:
        colnumber=len(Bconfig)
        pairs = np.array(list(product(Aconfig, Bconfig)))
        find_p = (pairs[:, 0] & pairs[:, 1]) == 0
        id_p = find_p.nonzero()[0]
        id_x=id_p//colnumber
        id_y=id_p%colnumber+count
        pair_index.extend([id_x,id_y])
        allowed_pairs = pairs[id_p]
        allowed_config = allowed_pairs[:, 0] + allowed_pairs[:, 1]
        Allowed_configs.append(allowed_config)
        count += colnumber
    del pairs
    Allowed_configs = np.concatenate(Allowed_configs) if Allowed_configs else np.array([], dtype=np.int64)
    return Allowed_configs,pair_index

if __name__== '__main__':
    Nx= 3
    Ny = 6
    Ns0 = Nx * Ny

    Nparticles = 6
    NA = 3
    NB = Nparticles - NA

    total_configs=np.array(bitstring_config(Ns0, Nparticles))
    klist = sample_klist(Nx, Ny)


    configsA = np.array(bitstring_config(Ns0, NA))
    KgroupsA = groupingKsum(configsA, klist, Nx, Ny)
    configsB = np.array(bitstring_config(Ns0, NB))
    KgroupsB = groupingKsum(configsB, klist, Nx, Ny)
    #Repeated_configs(KgroupsA[17], KgroupsB)
    solver = lambda Aconfig: Repeated_configs(Aconfig, KgroupsB)
    Pair= Parallel(n_jobs=8)(delayed(solver)(gid) for gid in KgroupsA)
    Paired_configs=[Pair[i][0] for i in range(Ns0)]
    Paired_pos=[Pair[i][1] for i in range(Ns0)]
    Paired_index=[]
    for i in range(Ns0):
        Paired_xid=np.concatenate([Paired_pos[i][j*2] for j in range(Ns0)])
        Paired_yid = np.concatenate([Paired_pos[i][j * 2+1] for j in range(Ns0)])
        Paired_index=np.vstack((Paired_xid,Paired_yid))
        np.save(f'Marix_index{i}.npy',Paired_index)
    del Paired_pos
    #print(Paired_configs)
    np.save('Allowed_pairs.npy', np.array(Paired_configs,dtype=object))
