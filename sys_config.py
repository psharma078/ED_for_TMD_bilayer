import numpy as np
from numpy import sqrt,pi 
from itertools import combinations

theta = 2.0
theta = theta*pi/180.0
ao = 3.52
aM = ao/theta

Lx = 3
Ly = 3
Nup_particles = 3

def moire_length():
    return aM

def system_size():
    return Lx, Ly

def particle_number():
    return Nup_particles

#bloch_states = np.arange(9)

def getspin(b,i):   #spin (0 or 1) at 'i' in the basis 'b'
    return (b>>i) & 1

def bitflip(b,i):  #Flips bits (1-->0, 0-->1) at loc 'i'
    return b^(1<<i)

def lcounter(b,i):  #left_counter: # of '1' in b left to site i
    num = b>>(i+1)
    return bin(num).count('1')


def fsign(b,imin,imax): #for two site C_i_sigma^dag C_j_sigma on spinful configs
    n1 = lcounter(b,imin)
    n2 = lcounter(b,imax)
    sign = 0
    if getspin(b,imax)==1:
        sign = n1 + n2 - 1
    else:
        sign = n1 + n2
    return (-1)**sign


def bitstring_config(sys_size, num_particle):

    decimal_numbers = []

    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ['0'] * sys_size
        for idx in ones_indices:
            binary_str[idx] = '1'
        decimal_numbers.append(int(''.join(binary_str), 2))

    return decimal_numbers

#For k<l: Ck Cl = -ak F_k+1 ... F_l-1 al 
#For i<j: Ci^dag Cj^dag = ai^dag F_i+1 ... F_j-1 aj^dag
#indices order [k1,k3,k2,k4]
def fermionic_signs(config,indices): #for spinless config Ck1dag Ck3dag Ck2 Ck4
    i = min(indices[0],indices[2])
    j = max(indices[0],indices[2])
    k = min([indices[1],indices[3]])
    l = max([indices[1],indices[3]])
    Fkl = np.array([getspin(config,x) for x in range(k+1,l)])
    Fkl = np.prod(1-2*Fkl)
    new_config = bitflip(bitflip(config,k),l)
    Fij = np.array([getspin(new_config,y) for y in range(i+1,j)])
    Fij = np.prod(1-2*Fij)
    new_config = bitflip(bitflip(new_config,j),i)
    fsign = 1
    if indices[1] < indices[3]:
        fsign *= -Fkl 
    else:
        fsign *= Fkl

    if indices[0] < indices[2]:
        fsign *= Fij
    else:
        fsign *= -Fij

    return fsign, new_config

    

if __name__ == "__main__":
    N=8
    u = 4
    d = 4
    up = bitstring_config(N,u)
    dn = bitstring_config(N,d)
    #for t in up:
    #    print(bin(t)[2:].zfill(N))
    #print(len(up))
    #print(up[10], bin(up[10])[2:].zfill(N))
    #print(up[14], bin(dn[14])[2:].zfill(N))
    #print(fsign(up[10],2,6), fsign(dn[14],2,7))

    con = 1234
    ind = [5,6,9,1]
    print(ind)
    print(bin(con)[2:].zfill(9))
    f,nin=fermionic_signs(con,ind)
    print(f)
    print(bin(nin)[2:].zfill(11))

"""
n_qubits = 3
hilbert_dimension = 2**n_qubits

for state_index in range(hilbert_dimension):
    state_bitstring = bin(state_index)[2:].zfill(n_qubits)
    #print(state_bitstring)
"""
