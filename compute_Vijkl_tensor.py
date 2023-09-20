import bilayerTBM
import sys_config
import numpy as np
from numpy import sqrt, pi, cos, sin
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

Lx, Ly = sys_config.system_size()
#########################################################
### non-interacting Eigenvalues and Eigenvectors ###
#########################################################
Htop,Hbot,Htunn = bilayerTBM.V_n_T_terms()

Q_vecs = bilayerTBM.Gvecs()
Q1 = Q_vecs[0]
Q2 = Q_vecs[1]

def crystal_momentum(Lx,Ly):#In the unit of Q1 and Q2
    xlist = np.linspace(0,1,Lx+1)
    xlist = xlist[:-1]
    ylist = np.linspace(0,1,Ly+1)
    ylist = ylist[:-1]
    k12list = np.array([np.array([x1,y1]) for x1 in xlist for y1 in ylist])
    return k12list

kmesh = crystal_momentum(Lx,Ly)
kbz_coord = kmesh @ np.array([Q1,Q2])
eigenSpectra = bilayerTBM.GS_Espectrum(kbz_coord,Htop,Hbot,Htunn)
del Htop, Hbot, Htunn
############################################################

g_list =  bilayerTBM.make_g_list() #in the unit of Q1 and Q2

def map_to_mBZ_momenta(q):  #q = [q] + gq (in the unit of Q1 and Q2)
    projQ1 = q[:,0]
    projQ2 = q[:,1]
    IQ1 = projQ1.astype(int)
    IQ2 = projQ2.astype(int)
    precession = 1e-6
    #FQ1 = np.round((projQ1 - IQ1)/precession)*precession
    #FQ2 = np.round((projQ2 - IQ2)/precession)*precession
    
    FQ1 = projQ1 - IQ1
    FQ2 = projQ2 - IQ2
    for j in range(len(FQ1)):
        if int(round(FQ1[j]*Lx*Ly)) == int(Lx*Ly):
            #print(FQ1[j], projQ1[j])
            FQ1[j] -= 1.0
            IQ1[j] += 1.0
        if int(round(FQ2[j]*Lx*Ly)) == int(Lx*Ly):
            FQ2[j] -= 1.0
            IQ2[j] += 1.0

        if FQ1[j] < -1e-6:
            #print(FQ1[j], projQ1[j])
            FQ1[j] += 1.0
            IQ1[j] -= 1.0
        if  FQ2[j] < -1e-6:
            FQ2[j] += 1.0
            IQ2[j] -= 1.0
    
    gq = np.column_stack((IQ1,IQ2))
    qmBZ = np.column_stack((FQ1,FQ2))

    return qmBZ, gq  #returns in the unit of Q1, Q2
    

EElist = np.array([eigenSpectra[i][0] for i in range(len(kmesh))])
EElist = EElist.flatten()
EVlist = np.array([eigenSpectra[i][1] for i in range(len(kmesh))])
EVlist = EVlist.reshape([len(kmesh),len(g_list),2],order='F')

q_list = np.array([k+g for k in kmesh for g in g_list]) #in terms of Q1, Q2

#print(q_list)
def matching_index(arr1, arr2, tolerance=1e-6):
    abs_diff = np.abs(arr1[:, np.newaxis] - arr2)
    matching_mask = np.all(abs_diff < tolerance, axis=2)
    matching_indices = np.where(matching_mask)
    return matching_indices

def sumZstrZ(k_indxList,g_indxList):
    Nmatch = len(k_indxList[0])
    g_indxList = [g_indxList[i] for i in k_indxList[1]] #arrange glistID according to [k3+q]ID
    solve = lambda j: sum(np.sum(EVlist[k_indxList[0][j],g_indxList[j][0],:].conjugate()\
            *EVlist[k_indxList[1][j],g_indxList[j][1],:],axis=0))
    ZstrZ_sum_gl = Parallel(n_jobs=num_cores)(delayed(solve)(j) for j in range(Nmatch))
    return np.array(ZstrZ_sum_gl)    

def find_gindx(g_kq): #find maching index for g and g(k3+q)+g'
    g_tot_list = np.array([gkq + gp for gkq in g_kq for gp in g_list])
    g_tot_list = g_tot_list.reshape([len(g_kq),len(g_list),2])
    solver = lambda g: matching_index(g_list,g)
    gindx_list = Parallel(n_jobs=num_cores)(delayed(solver)(g) for g in g_tot_list)
    return gindx_list

def F_kgl(q):
    #computing F([k3+q],k3,g_k3+q,sigma)
    k3_plus_q_list = kmesh + q
    k3_plus_q_mBZ, g_k3q = map_to_mBZ_momenta(k3_plus_q_list)
    k1_k3_indx_list = matching_index(kmesh,k3_plus_q_mBZ)
    k1=k1_k3_indx_list[0]
    k3=k1_k3_indx_list[1]
    #print(kmesh[k1]-k3_plus_q_mBZ[k3])
    #kk = k3_plus_q_mBZ @ np.array([Q1,Q2])
    #plt.scatter(kk[:,0],kk[:,1], color='r')
    g_indx_list = find_gindx(g_k3q) #where in the g_List g_(k3+q)+g' belong to
    Zk1Zk3_list = sumZstrZ(k1_k3_indx_list,g_indx_list) #Note k1 in k1_indx and k3 in kmesh
    
    #print(sum(np.sum(EVlist[3,g_indx_list[0][0],:].conjugate()*EVlist[0,g_indx_list[0][1],:],axis=0)))
    
    #computing F([k4-q],k4,g_k4-q,sigma)
    k4_minus_q_list = kmesh - q
    k4_minus_q_mBZ, g_k4q = map_to_mBZ_momenta(k4_minus_q_list)
    k2_k4_indx_list = matching_index(kmesh,k4_minus_q_mBZ)
    #kk = k4_minus_q_mBZ @ np.array([Q1,Q2])
    #plt.scatter(kk[:,0],kk[:,1], color='b')

    g_indx_list = find_gindx(g_k4q)
    Zk2Zk4_list = sumZstrZ(k2_k4_indx_list,g_indx_list)
    return k1_k3_indx_list, Zk1Zk3_list, k2_k4_indx_list, Zk2Zk4_list

"""
plt.plot([0,Q1[0]],[0,Q1[1]],color='k')
plt.plot([0,Q2[0]],[0,Q2[1]],color='k')
plt.plot([Q2[0],Q1[0]+Q2[0]],[Q2[1],Q2[1]],color='k')
plt.plot([Q1[0],Q1[0]+Q2[0]],[Q1[1],Q2[1]],color='k')
kmesh = kmesh @ np.array([Q1,Q2])
plt.scatter(kmesh[:,0],kmesh[:,1],color="green")
plt.axis("equal")
plt.show()
"""
#print(q_list[-2])
#F_kgl(q_list[-4])
#import sys
#sys.exit()

#########################################################################
###### computing tensor V(k1,k2,k3,k4) ##################################
#########################################################################
Nsys = Lx*Ly
aM = sys_config.moire_length()
Area = sqrt(3)/2.0 * Nsys*aM*aM*1e-8 #Angstrom*cm
#solveF = lambda q: F_kgl(q)
#result = Parallel(n_jobs=num_cores)(delayed(solveF)(q) for q in [q_list[0],q_list[-1]])

def Vk1k2k3k4(qlist):
    k1324 = {}
    Vinteraction = []
    key = 0
    epsilon = 10
    e =  4.8032046e-10 #statcoulomb
    erg_to_meV = 6.241506e+14

    for q in qlist:
        V_q = 0
        q_norm = np.linalg.norm(q @ np.array([Q1,Q2])) #per Angstrom
        if q_norm == 0.0:
            V_q = 0
        else:
            V_q = 2*pi*e*e/(q_norm*Area*epsilon)*erg_to_meV
        
        k1k3_idx,Fk1k3,k2k4_idx,Fk2k4 = F_kgl(q)
        k1k3_idx = np.column_stack((k1k3_idx[0],k1k3_idx[1]))
        k2k4_idx = np.column_stack((k2k4_idx[0],k2k4_idx[1]))
        Fk13 = {}
        Fk24 = {}
         
        for j, k13id in enumerate(k1k3_idx):
            Fk13[tuple(k13id)] = Fk1k3[j]
        
        for i, k24id in enumerate(k2k4_idx):
            Fk24[tuple(k24id)] = Fk2k4[i]
        
        shape_k13 = k1k3_idx.shape
        shape_k24 = k2k4_idx.shape
        k1324_idx = np.empty((shape_k13[0] * shape_k24[0], 4))

        k1324_idx[:,0:2] = (k1k3_idx[:, 0:2][:, np.newaxis,:]*np.ones((shape_k13[0], shape_k24[0],2))).reshape(-1,2)
        k1324_idx[:,2:] = (k2k4_idx[:, 0:2][np.newaxis,:,:]*np.ones((shape_k13[0], shape_k24[0],2))).reshape(-1,2)
        
        for k_id in k1324_idx:
            if k_id[0]==k_id[2] or k_id[1]==k_id[3]:
                continue

            k13 = tuple(k_id[:2])
            k24 = tuple(k_id[2:])
            Vint = V_q * Fk13[k13] * Fk24[k24]
            
            if abs(Vint) < 1e-6: continue

            row = tuple(k_id.astype(int))
            if row not in k1324:
                k1324[row] = key
                Vinteraction.append(Vint)
                key += 1
                
            else:
                loc = k1324[row]
                Vinteraction[loc] += Vint
    k1324 = np.array(list(k1324.keys()))

    return k1324, Vinteraction 

#k1324_id_list, V1234 = Vk1k2k3k4(q_list)
#print(k1324_id_list)
#print(V1234)
#lo = k1324_id_list[tuple(np.array([3,4,5,6]))]
#print(V1234[lo])

################################################################
############## using total momentum conservation ###############
################################################################
Nup = int(sys_config.particle_number())
up_configs = sys_config.bitstring_config(Nsys,Nup)
up_configs = np.array(up_configs)

def groupingKsum(configs,klist):
    dimH = len(configs)
    configs_bin = list(map(lambda x: format(x, '0'+str(Nsys)+'b'), configs))
    binary_lists = np.array([[int(bit) for bit in binary] for binary in configs_bin])  #convert binary num to array
    occ_orbitals = np.array([klist * binary_lists[j,:,np.newaxis] for j in range(dimH)])
    #occ_ksum = [np.sum(occ_orbitals[n,:],axis=0) for n in range(dimH)]
    occ_ksum = lambda w: np.sum(occ_orbitals[w,:],axis=0)
    occ_ksum = Parallel(n_jobs=num_cores)(delayed(occ_ksum)(w) for w in range(dimH))
    occ_ksum,_ = map_to_mBZ_momenta(np.array(occ_ksum))
    
    precision = 1e-6
    occ_ksum = np.round(occ_ksum / precision) * precision
    
    #bins_A = np.digitize(occ_ksum[:, 0], np.arange(0, 1 + precision, precision))
    #bins_B = np.digitize(occ_ksum[:, 1], np.arange(0, 1 + precision, precision))
    #df = pd.DataFrame({'A': bins_A, 'B': bins_B})
    df = pd.DataFrame(occ_ksum,columns=['A','B'])
    grouped_k = df.groupby(['A', 'B']).apply(lambda x: x.index.tolist()).tolist()

    #occ_orbitals = np.array([occ_orbitals[l,:] @ np.array([Q1,Q2]) for l in range(len(configs))])
    #occ_orb_coord = lambda l: occ_orbitals[l,:] @ np.array([Q1,Q2])
    #occ_orb_coord = Parallel(n_jobs=num_cores)(delayed(occ_orb_coord)(l) for l in range(dimH))
    #kSum = lambda w: np.sum(occ_orb_coord[w],axis=0)
    #kSum = Parallel(n_jobs=num_cores)(delayed(kSum)(w) for w in range(dimH))

    #df = pd.DataFrame(kSum,columns=['A','B'])
    #grouped_k = df.groupby(['A', 'B']).apply(lambda x: x.index.tolist()).tolist()
    #lt = 0
    #for i in range(len(grouped_k)):
    #    lt += len(grouped_k[i])
    #    print(grouped_k[i][0], occ_ksum[grouped_k[i][0]])
    return grouped_k

#print(len(groupingKsum(up_configs,kmesh)))
#import sys
#sys.exit()
#########################################################################
####### Setting up many body Hamiltonian for given lattice geometry #####
#########################################################################
from sys_config import getspin, bitflip, fermionic_signs

k1324_id_list, V1234 = Vk1k2k3k4(q_list)

def get_matrixEle_newconfig(config):
    Matel_newconfig = []
    for j, kindx in enumerate(k1324_id_list):
        k1id = kindx[0]
        k3id = kindx[1]
        k2id = kindx[2]
        k4id = kindx[3]
        if (getspin(config,k1id)+getspin(config,k2id)==0 and  getspin(config,k3id)*getspin(config,k4id)==1):
            fsign, newconfig = fermionic_signs(config,kindx)
            matel = 0.5 * fsign * V1234[j]  
            Matel_newconfig.append([matel,newconfig])
    return Matel_newconfig  

#Hamiltonian corresponding to specific K_tot configs
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

def diag_part_Ham(config):
    occ_orb = np.array([getspin(config,i) for i in range(Nsys)])
    diag_value = sum(EElist * occ_orb)
    return diag_value

def Hamiltonian(configs_indx):
    dimHam = len(configs_indx)
    #Ham = np.zeros((dimHam,dimHam), dtype=complex)
    rows = np.array([d for d in range(dimHam)])
    cols = np.array([d for d in range(dimHam)])
    
    configs = up_configs[configs_indx]
    mat_ele = np.array([diag_part_Ham(config) for config in configs])
    
    config_lookup = {value: index for index, value in enumerate(configs)}
    
    solver = lambda config: get_matrixEle_newconfig(config)
    matel_newconfig_list = Parallel(n_jobs=num_cores)(delayed(solver)(config) for config in configs)

    for i, matel_config in enumerate(matel_newconfig_list):
        matel_config = np.array(matel_config)
        configs_list = matel_config[:,1].real.astype(int)
        row = np.array([config_lookup[config] for config in configs_list])
        col = np.full(len(row), i)
        
        rows = np.concatenate((rows, row))
        cols = np.concatenate((cols, col))
        mat_ele = np.concatenate((mat_ele, matel_config[:,0]))
    
    matrix = sp.csc_matrix((mat_ele, (rows,cols)))

    shape = (dimHam,dimHam)
    #if (shape != matrix.shape):
    #    raise ValueError("Dimension of the Hamiltonian do not match with configuration space")
    #H = matrix.toarray()
    #print("Hermitian:", is_hermitian(matrix.toarray())) 
    Eigs, _ = eigs(matrix,k=dimHam-5,which='SM',sigma=None)
    #eigs, vecs = np.linalg.eigh(matrix.toarray())
    Eigs = np.sort(Eigs.real)
    print(Eigs)
    return Eigs

configsGroupID = groupingKsum(up_configs,kmesh)
#Hamiltonian(configsGroupID[-1])
#ngroup = len(configsGroupID)
#solver = lambda gid: Hamiltonian(gid)
#Evals = Parallel(n_jobs=num_cores)(delayed(solver)(gid) for gid in configsGroupID)
Evals = []
for gid in configsGroupID:
    Evals.append(Hamiltonian(gid))

del k1324_id_list, V1234
#############################################################################
#print(kmesh)
#print(configsID_group)
############################### PLOT EIGEN STATES ##########################

def plot_energies():
    N1 = Lx
    k1 = kmesh[:,0]
    k2 = kmesh[:,1]
    xvals = k1 + N1 * k2
    Egs = np.min(np.concatenate(Evals))
    #print("ground state=", Egs)
    fig,ax =  plt.subplots(figsize=(4,3))
    for i, eig in enumerate(Evals):
        x = np.full(len(eig), xvals[i])
        ax.scatter(x, np.array(eig)-Egs, c='blue')
    ax.set_xlabel("$k_1 + N_1 k_2$")
    ax.set_ylabel("$E-E_{GS}$")
    ax.set_ylim([-0.9,50])
    plt.savefig("Energy_N=5x3_nup=3.pdf",dpi=300,bbox_inches='tight')
    return fig, ax

fig, ax = plot_energies()
plt.show()
if __name__ == "__main__":
    a,b,c,d = F_kgl(q_list[1])
    #print(b)
    #print(a)
    #sumk = np.array([k1+k2 for k1 in kmesh for k2 in kmesh])
    #k,g = map_to_mBZ_momenta(sumk)
    #k, g = map_to_mBZ_momenta(q_list)

    #k = k @ np.array([Q1,Q2])
    #plt.scatter(k[:,0],k[:,1], color="r")

    #plt.plot([0,Q1[0]],[0,Q1[1]],color='k')
    #plt.plot([0,Q2[0]],[0,Q2[1]],color='k')
    #plt.plot([Q2[0],Q1[0]+Q2[0]],[Q2[1],Q2[1]],color='k')
    #plt.plot([Q1[0],Q1[0]+Q2[0]],[Q1[1],Q2[1]],color='k') 
    #kmesh = kmesh @ np.array([Q1,Q2])
    #plt.scatter(kmesh[:,0],kmesh[:,1],color="green")
    #plt.axis("equal")

    #km = bilayerTBM.bz_sampling(4,Q1,Q2)
    #plt.scatter(km[:,0],km[:,1], color="pink")
    #plt.show()



