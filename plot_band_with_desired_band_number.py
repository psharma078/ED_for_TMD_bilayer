import bilayerTBM
import numpy as np
from numpy import sqrt, pi, cos, sin,exp
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

ao = 3.52
theta = 4.4*pi/180
aM = ao/theta

Q_vecs = bilayerTBM.Gvecs(aM)
Q1 = Q_vecs[0]
Q2 = Q_vecs[1]

kappa_p = (Q1+Q2)/3.0
kappa_m = (2*Q1-Q2)/3.0


def special_kpath(npts):
    gamma = np.array([0,0])
    m = Q1/2.
    kline1 = bilayerTBM.kgrids(gamma,m,npts)
    kline2 = bilayerTBM.kgrids(m,kappa_p,npts)
    kline3 = bilayerTBM.kgrids(kappa_p,gamma,npts)
    klabels = [r"$\gamma$", r'$m$', r'$\kappa_+$', r'$\gamma$']
    k_dist = np.array([0,sqrt(sum((m-gamma)**2)),sqrt(sum((kappa_p-m)**2))\
            , sqrt(sum((kappa_p-gamma)**2))])
    kcoords = []
    for i in range(len(k_dist)):
        kcoords.append(sum(k_dist[:i+1]))
    klines = np.concatenate((kline1[:-1,:],kline2[:-1,:],kline3[:,:]))
    return klines, klabels, kcoords

def plot_bands(EElist,klist,kcoord,klabels,energy_range=[-200,100]):
    fig,ax = plt.subplots(figsize=(4,3))
    ksep = np.array([sqrt(sum((klist[l+1]-klist[l])**2)) for l in \
        range(klist.shape[0]-1)])
    ksep = np.insert(ksep,0,0)
    kline = []
    for i in range(len(ksep)):
        kline.append(sum(ksep[:i+1]))

    ax.plot(kline,EElist,'k',linewidth=1)
    ax.set_ylabel(r'$E/\mathrm{eV}$', fontsize=14)
    ax.set_ylim(energy_range)
    ax.set_xlim([0,max(kline)])
    ax.set_xticks(kcoord)
    ax.set_xticklabels(klabels,fontsize=14)
    return fig,ax

def bandStructure():
    npts = 40
    EnergyRange= [-150, 40]
    klist, klabels, kcoords = special_kpath(npts)
    Espectra = bilayerTBM.main(klist,1)
    EElist = [Espectra[i][0] for i in range(len(klist))]
    fig, ax = plot_bands(EElist, klist, kcoords, klabels, EnergyRange)
    ax.axvline(x=kcoords[1],linestyle="--",linewidth=0.8,c='purple')
    ax.axvline(x=kcoords[2],linestyle="--",linewidth=0.8,c='purple')
    return fig, ax

fig1, ax1 = bandStructure()

def electron_density(nbz,band_id):
    ksample = bilayerTBM.bz_sampling(nbz,Q1,Q2)
    nx = 100
    ny = 100
    xlist = np.linspace(-100,100,nx)
    ylist = np.linspace(-100,100,ny)
    xx,yy = np.meshgrid(xlist,ylist)
    x = xx.flatten()
    y = yy.flatten()
    g_list = bilayerTBM.make_g_list(1,Q1,Q2)
    qxList = np.array([G[0] for G in g_list])
    qyList = np.array([G[1] for G in g_list])
    spectra = bilayerTBM.main(ksample,1)
    EvecList = np.array([spectra[i][1] for i in range(len(ksample))])
    expfac = 1j*(qxList[:,np.newaxis]*x[np.newaxis,:] + qyList[:,np.newaxis]*y[np.newaxis,:])
    density = np.zeros((nx,ny))
    halfdim = len(g_list)
    for Evec in EvecList:
        Evec = Evec[:,-band_id] #from the highest eigenvalue
        wf_top = np.dot(Evec[:halfdim],exp(expfac))
        wf_bot = np.dot(Evec[halfdim:],exp(expfac))
        wf_top = np.reshape(wf_top,(nx,ny))
        wf_bot = np.reshape(wf_bot,(nx,ny))
        density += np.abs(wf_top)**2
        density += np.abs(wf_bot)**2

    #plot density
    cmap = 'hot'#'RdBu'
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
    #ax.set_title('Top Layer')
    c = ax.pcolormesh(xx/aM,yy/aM,density,shading = 'gouraud',cmap = cmap)
    cbar = plt.colorbar(c, label='density')
    return fig,ax
fig2, ax2 = electron_density(nbz=20, band_id=1)
plt.show()
