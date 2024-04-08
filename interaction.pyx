cimport cython
from cython.parallel cimport parallel, prange
cimport numpy as cnp
from operator import itemgetter
cdef extern int __builtin_popcountll(unsigned long long) nogil


cnp.import_array()

ctypedef unsigned long long ull
ctypedef cnp.int64_t Dtype_int64_t
ctypedef cnp.complex128_t Dtype_complex128_t
ctypedef cnp.float64_t Dtype_float64_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

cdef ull next_subset(ull subset):
   cdef ull smallest, ripple, ones;
   smallest = subset& -subset;
   ripple = subset + smallest;    
   ones = subset ^ ripple;
   ones = (ones >> 2)//smallest;
   subset= ripple | ones;
   return subset


cdef list subset2list(ull subset, int offset, int cnt):  
    cdef list lst=[0]*cnt #pre-allocate
    cdef int current=0;
    cdef int index=0;
    while subset>0:
        if((subset&1)!=0):
            lst[index]=offset+current;
            index+=1;
        subset>>=1;
        current+=1;
    return lst

def bitstring_config(int n, int k):
    cdef ull MAX=1L<<n;
    cdef ull subset=(1L<<k)-1L;
    cdef int ii;
    cdef ull num;
    cdef list lst=[0]*k;
    cdef list decimals=[];
    cdef list subsets=[];
    while(MAX>subset):
        num = 0;
        lst = subset2list(subset, 0, k);
        for ii in lst:
            num += (1<<ii);
        decimals.append(num);
        subsets.append(lst);
        subset=next_subset(subset);
    return decimals, subsets

cdef int getspin(ull b,int i):   #spin (0 or 1) at 'i' in the basis 'b'
    return (b>>i) & 1

cdef ull bitflip(ull b,int i):  #Flips bits (1-->0, 0-->1) at loc 'i'
    return b^(1<<i)

cdef int lcounter(ull b,int i):  #left_counter: # of '1' in b left to site i
    cdef ull num;
    num = b>>(i+1);
    return __builtin_popcountll(num)

cdef c_an(ull configs, int site):
    cdef ull newconfigs=bitflip(configs, site);
    cdef int fsign = 1-2*(lcounter(configs, site)%2);
    return fsign, newconfigs

cdef c_dag(ull configs, int site):
    cdef ull newconfigs = bitflip(configs, site);
    cdef int fsign = 1-2*(lcounter(configs, site)%2);
    return fsign, newconfigs

def get_matrixEle(tuple configs, list k1234, list V1234):
    cdef cnp.ndarray[Dtype_int64_t,ndim=1] ks;
    cdef Dtype_complex128_t V;
    cdef list configs_k3, configs_k4;
    cdef int k1,k2,k3,k4,f1, f2, f3,f4,fsign, jj;
    cdef int nbasis = len(configs);
    cdef ull config, new_config;
    cdef list row=[];
    cdef list Matele=[];
    cdef list col=[];
    cdef dict config_lookup = {value: index for index, value in enumerate(configs)}
    cdef int newconfig_id

    for ks,V in zip(k1234,V1234):
        k1,k2,k3,k4 =ks[0], ks[1],ks[2], ks[3] 
        for jj in range(nbasis):
            if (getspin(configs[jj], k3)==1):
                f3, new_config = c_an(configs[jj], k3);
            else:
                continue;
            if (getspin(new_config,k4)==1):
                f4,new_config = c_an(new_config,k4)
            else: 
                continue;
            if (getspin(new_config,k2)==0):
                f2, new_config = c_dag(new_config, k2)
            else:
                continue;
            if (getspin(new_config, k1) == 0):
                f1, new_config = c_dag(new_config, k1);
                fsign = f1 * f2 * f3 * f4
            else: 
                continue;
            col.append(jj)
            newconfig_id = config_lookup[new_config]
            row.append(newconfig_id)
            Matele.append(fsign * V)
    return row, col, Matele

def get_matrixEle_spinful(tuple configs, int Nsys, list k1234_upup, list V1234_upup, list k1234_dndn, list V1234_dndn, list k1234_updn, list V1234_updn, list k1234_dnup, list V1234_dnup):
# Note that configs are (configup, configdn), which can be converted to a unique integer index (configup << Nsys) + configdn
    cdef cnp.ndarray[Dtype_int64_t,ndim=1] ks;
    cdef Dtype_complex128_t V;
    cdef list configs_k3, configs_k4;
    cdef int k1,k2,k3,k4, f1, f2, f3,f4,fsign, jj, ii;
    cdef int nbasis = len(configs);
    cdef ull new_config_up, new_config_dn;
    cdef list row=[];
    cdef list Matele=[];
    cdef list col=[];
    cdef dict config_lookup = {(value[0]<<Nsys)+value[1]: index for index, value in enumerate(configs)}
    cdef int newconfig_id;

    # k1up k2up k4up k3up
    for ks,V in zip(k1234_upup,V1234_upup):
        k1,k2,k3,k4 =ks[0], ks[1],ks[2], ks[3] 
        for jj in range(nbasis):
            configup = configs[jj][0]
            if (getspin(configup, k3)==1):
                f3, new_config_up = c_an(configup, k3);
            else:
                continue;
            if (getspin(new_config_up,k4)==1):
                f4, new_config_up = c_an(new_config_up,k4)
            else: 
                continue;
            if (getspin(new_config_up,k2)==0):
                f2, new_config_up = c_dag(new_config_up, k2)
            else:
                continue;
            if (getspin(new_config_up, k1) == 0):
                f1, new_config_up = c_dag(new_config_up, k1);
                fsign = f1 * f2 * f3 * f4
            else: 
                continue;
            new_config_dn = configs[jj][1]
            col.append(jj)
            newconfig_id = config_lookup[(new_config_up<<Nsys)+new_config_dn]
            row.append(newconfig_id)
            Matele.append(fsign * V)

    # k1dn k2dn k4dn k3dn
    for ks,V in zip(k1234_dndn,V1234_dndn):
        k1,k2,k3,k4 =ks[0], ks[1],ks[2], ks[3] 
        for jj in range(nbasis):
            configdn = configs[jj][1]
            if (getspin(configdn, k3)==1):
                f3, new_config_dn = c_an(configdn, k3);
            else:
                continue;
            if (getspin(new_config_dn,k4)==1):
                f4,new_config_dn = c_an(new_config_dn,k4)
            else: 
                continue;
            if (getspin(new_config_dn,k2)==0):
                f2, new_config_dn = c_dag(new_config_dn, k2)
            else:
                continue;
            if (getspin(new_config_dn, k1) == 0):
                f1, new_config_dn = c_dag(new_config_dn, k1);
                fsign = f1 * f2 * f3 * f4
            else: 
                continue;
            new_config_up = configs[jj][0]
            col.append(jj)
            newconfig_id = config_lookup[(new_config_up<<Nsys)+new_config_dn]
            row.append(newconfig_id)
            Matele.append(fsign * V)
   
    # k1dn k2up k4up k3dn
    for ks,V in zip(k1234_dnup,V1234_dnup):
        k1,k2,k3,k4 =ks[0], ks[1],ks[2], ks[3] 
        for jj in range(nbasis):
            configup = configs[jj][0]
            configdn = configs[jj][1]
            if (getspin(configdn, k3)==1):
                f3, new_config_dn = c_an(configdn,k3);
            else:
                continue;
            if (getspin(configup,k4)==1):
                f4, new_config_up = c_an(configup,k4)
            else: 
                continue;
            if (getspin(new_config_up,k2)==0):
                f2, new_config_up = c_dag(new_config_up, k2)
            else:
                continue;
            if (getspin(new_config_dn, k1) == 0):
                f1, new_config_dn = c_dag(new_config_dn, k1);
                fsign = f1 * f2 * f3 * f4
            else: 
                continue;
            col.append(jj)
            newconfig_id = config_lookup[(new_config_up<<Nsys)+new_config_dn]
            row.append(newconfig_id)
            Matele.append(fsign * V)
    # k1up k2dn k4dn k3up
    for ks,V in zip(k1234_updn,V1234_updn):
        k1,k2,k3,k4 =ks[0], ks[1],ks[2], ks[3] 
        for jj in range(nbasis):
            configup = configs[jj][0]
            configdn = configs[jj][1]
            if (getspin(configup,k3)==1):
                f3, new_config_up = c_an(configup, k3);
            else:
                continue;
            if (getspin(configdn,k4)==1):
                f4, new_config_dn = c_an(configdn,k4)
            else: 
                continue;
            if (getspin(new_config_dn,k2)==0):
                f2, new_config_dn = c_dag(new_config_dn, k2)
            else:
                continue;
            if (getspin(new_config_up, k1) == 0):
                f1, new_config_up = c_dag(new_config_up, k1);
                fsign = f1 * f2 * f3 * f4
            else: 
                continue;
            col.append(jj)
            newconfig_id = config_lookup[(new_config_up<<Nsys)+new_config_dn]
            row.append(newconfig_id)
            Matele.append(fsign * V)
    return row, col, Matele

 
