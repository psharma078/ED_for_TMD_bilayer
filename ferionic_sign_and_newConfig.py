#For i<j: Ci Cj = -ai F_i+1 ... F_j-1 aj
#For k<l: Ck^dag Cl^dag = ak^dag F_k+1 ... F_l-1 al^dag
#indices order [k1,k2,k3,k4]
def fermionic_signs(config,indices): #for spinless config Ck4dag Ck3dag Ck1 Ck2
    i = min(indices[0],indices[1])
    j = max(indices[0],indices[1])
    k = min([indices[2],indices[3]])
    l = max([indices[2],indices[3]])
    Fij = np.array([getspin(config,x) for x in range(i+1,j)])
    Fij = np.prod(1-2*Fij)
    new_config = bitflip(bitflip(config,i),j)
    Fkl = np.array([getspin(new_config,y) for y in range(k+1,l)])
    Fkl = np.prod(1-2*Fkl)
    new_config = bitflip(bitflip(new_config,k),l)
    fsign = 1
    if indices[3] < indices[2]:
        fsign *= Fkl
    else:
        fsign *= -Fkl

    if indices[0] < indices[1]:
        fsign *= -Fij
    else:
        fsign *= Fij

    return fsign, new_config


def get_matrixEle_newconfig(config):
    Matel_newconfig = []
    for j, kindx in enumerate(k1324_id_list):
        k1id = kindx[0]
        k2id = kindx[1]
        k3id = kindx[2]
        k4id = kindx[3]
        if (getspin(config,k3id)+getspin(config,k4id)==0 and  getspin(config,k1id)*getspin(config,k2id)==1):
            fsign, newconfig = fermionic_signs(config,kindx)
            matel = 0.5 * fsign * V1234[j]
            Matel_newconfig.append([matel,newconfig])
    return Matel_newconfig