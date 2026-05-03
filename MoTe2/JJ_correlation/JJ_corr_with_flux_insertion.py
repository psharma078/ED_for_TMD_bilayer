import argparse
import multiprocessing
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numba import njit
from numba.typed import List
from numpy import cos, pi, sin, sqrt
from scipy.sparse.linalg import eigsh


num_cores = multiprocessing.cpu_count()
epsilon_r = 10.0


def main(theta_deg, N1, N2, Nup, Vd, ntheta=14, gs_ktot_ind=(0,), output="JJcorr_TBC_minimal.npz", n_jobs=None):
    if n_jobs is None or n_jobs <= 0:
        n_jobs = num_cores
    theta = theta_deg / 180.0 * pi
    a0 = 3.52e-10
    gvecs = g_vecs(theta, a0)
    Q1, Q2 = reciprocal_vecs(gvecs)
    aM = a0 / (2 * np.sin(theta / 2))
    kappa_p = (2 * Q1 + Q2) / 3
    kappa_m = (Q1 - Q2) / 3

    Nk = 6
    n_k = N1 * N2
    n_g = (2 * Nk + 1) ** 2

    nk_vals = np.arange(-Nk, Nk + 1)
    ones = np.ones(2 * Nk + 1)
    mmlist = np.kron(nk_vals, ones)
    nnlist = np.kron(ones, nk_vals)
    Qxlist = nnlist * Q1[0] + mmlist * Q2[0]
    Qylist = nnlist * Q1[1] + mmlist * Q2[1]

    diagm = np.diag(np.ones(2 * Nk, dtype=np.complex128), -1)
    diagp = diagm.T
    diag_Nk = np.eye(2 * Nk + 1, dtype=np.complex128)
    delta_nn_mmp1 = np.kron(diagm, diag_Nk)
    delta_nnp1_mm = np.kron(diag_Nk, diagm)
    delta_nnm1_mmm1 = np.kron(diagp, diagp)
    delta_nn_mm = np.kron(diag_Nk, diag_Nk)
    delta_nn_mmm1 = delta_nn_mmp1.T

    halfdim = n_g
    ham_dim = 2 * halfdim

    def gen_layer_hamiltonian(k, dk):
        me = 9.10938356e-31
        mass = 0.62 * me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21
        prefactor = hbar**2 / (2 * mass) * J_to_meV

        kx_top = Qxlist + k[0] + dk[0] - kappa_p[0]
        ky_top = Qylist + k[1] + dk[1] - kappa_p[1]
        kx_bottom = Qxlist + k[0] + dk[0] - kappa_m[0]
        ky_bottom = Qylist + k[1] + dk[1] - kappa_m[1]

        psi = -91 / 180.0 * pi
        V = 11.2
        H_top = -V * (delta_nn_mmp1 + delta_nnp1_mm + delta_nnm1_mmm1) * np.exp(1j * psi)
        H_top += H_top.conj().T
        H_bottom = H_top.conj()

        np.fill_diagonal(H_top, (kx_top**2 + ky_top**2) * prefactor + Vd / 2.0)
        np.fill_diagonal(H_bottom, (kx_bottom**2 + ky_bottom**2) * prefactor - Vd / 2.0)
        return H_top, H_bottom

    def construct_ham(k, dk):
        H_top, H_bottom = gen_layer_hamiltonian(k, dk)
        w = 13.3
        Delta_T = -w * (delta_nn_mm + delta_nnm1_mmm1 + delta_nn_mmm1)

        ham = np.zeros((ham_dim, ham_dim), dtype=np.complex128)
        ham[:halfdim, :halfdim] = H_top
        ham[halfdim:, halfdim:] = H_bottom
        ham[:halfdim, halfdim:] = Delta_T.conj().T
        ham[halfdim:, :halfdim] = Delta_T
        return ham

    def H_der(k, dk):
        me = 9.10938356e-31
        mass = 0.62 * me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21
        prefactor = hbar**2 / (2 * mass) * J_to_meV

        kx_top = Qxlist + k[0] + dk[0] - kappa_p[0]
        ky_top = Qylist + k[1] + dk[1] - kappa_p[1]
        kx_bottom = Qxlist + k[0] + dk[0] - kappa_m[0]
        ky_bottom = Qylist + k[1] + dk[1] - kappa_m[1]

        ham_x = np.zeros((ham_dim, ham_dim), dtype=np.complex128)
        ham_y = np.zeros((ham_dim, ham_dim), dtype=np.complex128)
        np.fill_diagonal(ham_x[:halfdim, :halfdim], 2 * prefactor * kx_top)
        np.fill_diagonal(ham_y[:halfdim, :halfdim], 2 * prefactor * ky_top)
        np.fill_diagonal(ham_x[halfdim:, halfdim:], 2 * prefactor * kx_bottom)
        np.fill_diagonal(ham_y[halfdim:, halfdim:], 2 * prefactor * ky_bottom)
        return ham_x, ham_y

    def velocity_matrix_element(k, dk, psi0, psi1):
        ham_x, ham_y = H_der(k, dk)
        return psi0.conj() @ ham_x @ psi1, psi0.conj() @ ham_y @ psi1

    def compute_F_mat(klist, dk):
        kfrac = klist.astype(np.float64)
        kfrac[:, 0] /= N1
        kfrac[:, 1] /= N2
        kxylist = kfrac @ np.array([Q1, Q2])

        result = [eigh_sorted(construct_ham(k, dk)) for k in kxylist]

        dim = 2 * n_g
        single_vecs = np.array([result[j][1][:, -1] for j in range(n_k)])
        single_E = np.array([result[j][0] for j in range(n_k)]).reshape([N1, N2, dim], order="F")[:, :, -1]

        VV = np.array([result[j][1] for j in range(n_k)], dtype=np.complex128)
        VV = VV.reshape([N1, N2, 2 * Nk + 1, 2 * Nk + 1, 2, dim], order="F")[:, :, :, :, :, -1]

        Fmat = np.zeros([N1, N2, N1, N2, 2 * Nk + 1, 2 * Nk + 1], dtype=np.complex128)
        Fmat[:, :, :, :, Nk, Nk] = np.einsum("abmnl,cdmnl->abcd", VV.conj(), VV)

        for jj in range(1, Nk + 1):
            for ii in range(1, Nk + 1):
                Fmat[:, :, :, :, Nk + ii, Nk + jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, ii:, jj:, :].conj(), VV[:, :, :-ii, :-jj, :])
                Fmat[:, :, :, :, Nk + ii, Nk - jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, ii:, :-jj, :].conj(), VV[:, :, :-ii, jj:, :])
                Fmat[:, :, :, :, Nk - ii, Nk + jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, :-ii, jj:, :].conj(), VV[:, :, ii:, :-jj, :])
                Fmat[:, :, :, :, Nk - ii, Nk - jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, :-ii, :-jj, :].conj(), VV[:, :, ii:, jj:, :])

        for jj in range(1, Nk + 1):
            Fmat[:, :, :, :, Nk + jj, Nk] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, jj:, :, :].conj(), VV[:, :, :-jj, :, :])
            Fmat[:, :, :, :, Nk - jj, Nk] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, :-jj, :, :].conj(), VV[:, :, jj:, :, :])
            Fmat[:, :, :, :, Nk, Nk + jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, :, jj:, :].conj(), VV[:, :, :, :-jj, :])
            Fmat[:, :, :, :, Nk, Nk - jj] = np.einsum("abmnl,cdmnl->abcd", VV[:, :, :, :-jj, :].conj(), VV[:, :, :, jj:, :])

        Fmat = Fmat.reshape(n_k, n_k, n_g, order="F")
        return Fmat.flatten(order="F"), single_E, single_vecs

    @njit
    def find_id_in_klist(k):
        return k[:, 0] + N1 * k[:, 1]

    @njit
    def find_id_in_glist(g):
        g += Nk
        return g[:, 0] + (2 * Nk + 1) * g[:, 1]

    @njit
    def get_k_q_ids(k3_id, k4_id, klist, qlist):
        k3_q = klist[k3_id] + qlist
        k4_q = klist[k4_id] - qlist
        g3, k3_red = compute_reduce_k(k3_q, N1, N2)
        g4, k4_red = compute_reduce_k(k4_q, N1, N2)
        return find_id_in_klist(k3_red), find_id_in_glist(g3), find_id_in_klist(k4_red), find_id_in_glist(g4)

    @njit
    def compute_coulomb(k3_id, k4_id, klist, qlist, Fmat1, Fmat2, Vlist):
        k3_q_id, g3_id, k4_q_id, g4_id = get_k_q_ids(k3_id, k4_id, klist, qlist)
        keep = (g3_id < n_g) & (g3_id >= 0) & (g4_id < n_g) & (g4_id >= 0) & (k3_q_id < k4_q_id)
        k3_q_id = k3_q_id[keep]
        k4_q_id = k4_q_id[keep]
        V = Fmat1[g3_id[keep] * n_k**2 + k3_id * n_k + k3_q_id]
        V *= Fmat2[g4_id[keep] * n_k**2 + k4_id * n_k + k4_q_id] * Vlist[keep]

        k1k2list = np.vstack((k3_q_id, k4_q_id)).T
        k1k2, k1k2_ids = two_index_unique(k1k2list, n_k)
        Vk1k2 = List()
        for ids in k1k2_ids:
            Vk1k2.append(np.sum(V[ids]))
        return Vk1k2, k1k2

    def coulomb_potential(qlist):
        qfrac = qlist.astype(np.float64)
        qfrac[:, 0] /= N1
        qfrac[:, 1] /= N2
        qxy = qfrac @ np.array([Q1, Q2])
        absq = np.linalg.norm(qxy, axis=1)

        k0 = 8.99e9
        J_to_meV = 6.242e21
        e_charge = 1.602e-19
        area = np.sqrt(3) / 2 * N1 * N2 * aM**2
        Vc = np.zeros_like(absq)
        nonzero = absq > 1e-14
        Vc[nonzero] = 2 * np.pi * e_charge**2 / (epsilon_r * absq[nonzero]) * J_to_meV / area * k0
        return Vc

    @njit
    def create_table_from_k3k4(k3_id, k4_id, klist, qlist, Fmat1, Fmat2, Vlist):
        k1234 = List()
        V12, k12 = compute_coulomb(k3_id, k4_id, klist, qlist, Fmat1, Fmat2, Vlist)
        for kpair in k12:
            k1234.append(np.concatenate((kpair, np.array([k3_id, k4_id]))))
        return k1234, V12

    @njit
    def coulomb_matrix_elements(qlist, Vlist, Fmat):
        k1234_all = List()
        V_all = List()
        for ii in range(n_k):
            for jj in range(ii):
                _, Vtemp = create_table_from_k3k4(ii, jj, klist, qlist, Fmat, Fmat, Vlist)
                k1234, Vdirect = create_table_from_k3k4(jj, ii, klist, qlist, Fmat, Fmat, Vlist)
                k1234_all.append(k1234)

                Vsector = List()
                for ll in range(len(Vdirect)):
                    Vsector.append(2 * (Vdirect[ll] - Vtemp[ll]))
                V_all.append(Vsector)
        return k1234_all, V_all

    def create_coulomb_table(k1234_all, V_all):
        return np.vstack(k1234_all), np.hstack(V_all)

    def current_current_corr(qval, Evec, Uvecs, configs, klist, dk, Len):
        config_lookup = {value: index for index, value in enumerate(configs)}
        k_ind = find_id_in_klist(klist.copy())

        k2_m_q = klist - qval
        g2, k2_m_q_red = compute_reduce_k(k2_m_q, N1, N2)
        k2_m_q_id = find_id_in_klist(k2_m_q_red)
        g2_id = find_id_in_glist(g2.copy())

        k1_p_q = klist + qval
        g1, k1_p_q_red = compute_reduce_k(k1_p_q, N1, N2)
        k1_p_q_id = find_id_in_klist(k1_p_q_red)
        g1_id = find_id_in_glist(g1.copy())

        keep2 = (g2_id < n_g) & (g2_id >= 0)
        keep1 = (g1_id < n_g) & (g1_id >= 0)
        k2_m_q_id = k2_m_q_id[keep2]
        g2_id = g2_id[keep2]
        k1_p_q_id = k1_p_q_id[keep1]
        g1_id = g1_id[keep1]

        k1_ind = k_ind[keep1]
        k2_ind = k_ind[keep2]

        kxylist = klist.astype(np.float64)
        kxylist[:, 0] /= N1
        kxylist[:, 1] /= N2
        kxylist = kxylist @ np.array([Q1, Q2])
        qxy = (qval.astype(float) @ np.array([Q1, Q2])) / np.array([N1, N2])

        origin_id = int((n_g - 1) / 2)
        JJxx = 0.0
        JJxy = 0.0
        JJyx = 0.0
        JJyy = 0.0

        for i, k1 in enumerate(k1_ind):
            shift = origin_id - g1_id[i]
            u1 = Uvecs[k1_p_q_id[i]].copy()
            u1[:n_g] = np.roll(u1[:n_g], shift)
            u1[n_g:] = np.roll(u1[n_g:], shift)
            v1_x, v1_y = velocity_matrix_element(kxylist[k1] + qxy / 2.0, dk, u1, Uvecs[k1])
            v1_x, v1_y = 1e7 * v1_x, 1e7 * v1_y

            config_k1 = getspin(configs, k1)
            valid_k1 = (config_k1 == 1).nonzero()[0]
            f1, new_config = c_an(configs[valid_k1], k1)
            valid_k1pq = (getspin(new_config, k1_p_q_id[i]) == 0).nonzero()[0]
            f2, new_config = c_dag(new_config[valid_k1pq], k1_p_q_id[i])
            f2 = f1[valid_k1pq] * f2

            for j, k2 in enumerate(k2_ind):
                valid_k2 = (getspin(new_config, k2) == 1).nonzero()[0]
                f3, newconfig = c_an(new_config[valid_k2], k2)
                f3 = f2[valid_k2] * f3
                valid_k2mq = (getspin(newconfig, k2_m_q_id[j]) == 0).nonzero()[0]
                f4, newconfig = c_dag(newconfig[valid_k2mq], k2_m_q_id[j])
                fsign = f3[valid_k2mq] * f4

                old_ids = valid_k1[valid_k1pq[valid_k2[valid_k2mq]]]
                new_ids = np.array([config_lookup[nc] for nc in newconfig]).astype(int)

                shift = origin_id - g2_id[j]
                u2 = Uvecs[k2_m_q_id[j]].copy()
                u2[:n_g] = np.roll(u2[:n_g], shift)
                u2[n_g:] = np.roll(u2[n_g:], shift)
                v2_x, v2_y = velocity_matrix_element(kxylist[k2] - qxy / 2.0, dk, u2, Uvecs[k2])
                v2_x, v2_y = 1e7 * v2_x, 1e7 * v2_y

                if Len == 1:
                    for state in range(len(Evec[0])):
                        mb_term = np.sum(Evec[:, state][new_ids].conj() * Evec[:, state][old_ids] * fsign)
                        JJxx += v1_x * v2_x * mb_term
                        JJxy += v1_x * v2_y * mb_term
                        JJyx += v1_y * v2_x * mb_term
                        JJyy += v1_y * v2_y * mb_term
                else:
                    mb_term = np.sum(Evec[new_ids].conj() * Evec[old_ids] * fsign)
                    JJxx += v1_x * v2_x * mb_term
                    JJxy += v1_x * v2_y * mb_term
                    JJyx += v1_y * v2_x * mb_term
                    JJyy += v1_y * v2_y * mb_term

        return np.array([JJxx, JJxy, JJyx, JJyy])

    def compute_JJ_for_q(qval, Evecs, Uvecs, configs, klist, dk):
        Len = len(configs)
        if Len == 1:
            JJ = current_current_corr(qval, Evecs, Uvecs, configs[0], klist, dk, Len)
            return JJ.real / len(Evecs[0])

        JJ = 0.0
        for p in range(Len):
            JJ += current_current_corr(qval, Evecs[p], Uvecs, configs[p], klist, dk, Len)
        return JJ.real / Len

    klist, qlist = sample_k_q(N1, N2, Nk)
    Vlist = coulomb_potential(qlist)

    up_configs = np.array(bitstring_config(n_k, Nup))
    zero_E = np.zeros((N1, N2), dtype=np.float64)
    Ktot, groups, _ = groupingKsum(up_configs, klist, zero_E, N1, N2)

    gs_ktot_ind = list(gs_ktot_ind)
    config_groups = [groups[g] for g in gs_ktot_ind]
    configs_for_JJ = [up_configs[g] for g in config_groups]

    flux_vals = np.linspace(0, 1, ntheta, endpoint=False)
    flux_grid = np.array([[phi1, phi2] for phi1 in flux_vals for phi2 in flux_vals])
    dk_list = (flux_grid / np.array([N1, N2])) @ np.array([Q1, Q2])

    print("aM", aM)
    print("Ktot =", Ktot[gs_ktot_ind])
    print("flux grid size =", len(dk_list))
    print("parallel flux jobs =", n_jobs)

    def solve_flux(flux_id, dk):
        print("flux", flux_id + 1, "of", len(dk_list), "phi/2pi =", flux_grid[flux_id])

        Fmat, single_E, Uvecs = compute_F_mat(klist, dk)
        _, _, single_manybody_E = groupingKsum(up_configs, klist, single_E, N1, N2)
        k1234, V1234 = create_coulomb_table(*coulomb_matrix_elements(qlist, Vlist, Fmat))

        if len(config_groups) > 1:
            solver = lambda group: manybody_diagonalize(len(config_groups), group, up_configs, single_manybody_E, k1234, V1234)
            results = [solver(group) for group in config_groups]
            Evecs = [result[0] for result in results]
            eigs = np.sort(np.concatenate([result[1] for result in results]))[:3]
        else:
            Evecs, eigs = manybody_diagonalize(len(config_groups), config_groups[0], up_configs, single_manybody_E, k1234, V1234)
            eigs = np.sort(eigs)[:3]
        if len(eigs) < 3:
            eigs = np.pad(eigs, (0, 3 - len(eigs)), constant_values=np.nan)

        JJ = np.array([compute_JJ_for_q(q, Evecs, Uvecs, configs_for_JJ, klist, dk) for q in klist])
        return JJ, eigs

    results = Parallel(n_jobs=n_jobs)(delayed(solve_flux)(flux_id, dk) for flux_id, dk in enumerate(dk_list))
    JJ_by_flux = [result[0] for result in results]
    energies_by_flux = [result[1] for result in results]

    JJ_by_flux = np.array(JJ_by_flux)
    energies_by_flux = np.array(energies_by_flux)
    JJ_avg = JJ_by_flux.mean(axis=0)

    output_prefix = output.rsplit(".", 1)[0]
    plot_spectral_outputs(flux_vals, energies_by_flux, output_prefix)
    print_results(klist, flux_grid, energies_by_flux, JJ_by_flux, JJ_avg)

    np.savez(
        output,
        JJcorr_avg=JJ_avg,
        JJcorr_twist=JJ_by_flux,
        dtheta_grid=flux_grid,
        ktot=Ktot[gs_ktot_ind],
        mb_energies_twist=energies_by_flux,
    )
    print("saved", output)
    print("saved", output_prefix + "_spectral_phi2.png")
    print("saved", output_prefix + "_energy_heatmap.png")


def plot_spectral_outputs(flux_vals, energies_by_flux, output_prefix):
    ntheta = len(flux_vals)
    energy_grid = energies_by_flux.reshape(ntheta, ntheta, energies_by_flux.shape[1])
    emin = np.nanmin(energy_grid)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    phi2_cut = energy_grid[0, :, :] - emin
    for state in range(phi2_cut.shape[1]):
        ax.plot(flux_vals, phi2_cut[:, state], marker="o", ms=3, lw=1, label=f"state {state}")
    ax.set_xlabel(r"$\phi_2/2\pi$")
    ax.set_ylabel(r"$E-E_{\min}$ (meV)")
    ax.set_title(r"$\phi_1=0$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_prefix + "_spectral_phi2.png", dpi=200)
    plt.close(fig)

    nstates = energy_grid.shape[2]
    fig, axes = plt.subplots(1, nstates, figsize=(4.0 * nstates, 3.3), squeeze=False)
    for state in range(nstates):
        ax = axes[0, state]
        im = ax.imshow(
            (energy_grid[:, :, state] - emin).T,
            origin="lower",
            extent=(flux_vals[0], flux_vals[-1], flux_vals[0], flux_vals[-1]),
            aspect="auto",
        )
        ax.set_xlabel(r"$\phi_1/2\pi$")
        ax.set_ylabel(r"$\phi_2/2\pi$")
        ax.set_title(f"state {state}")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"$E-E_{\min}$ (meV)")
    fig.tight_layout()
    fig.savefig(output_prefix + "_energy_heatmap.png", dpi=200)
    plt.close(fig)


def print_results(klist, flux_grid, energies_by_flux, JJ_by_flux, JJ_avg):
    print()
    print("many-body energies from the JJ diagonalizations")
    print("columns: flux_index phi1/2pi phi2/2pi E0 E1 E2")
    for i, (flux, eigs) in enumerate(zip(flux_grid, energies_by_flux)):
        print(f"{i:4d} {flux[0]: .8f} {flux[1]: .8f} {eigs[0]: .8e} {eigs[1]: .8e} {eigs[2]: .8e}")

    print()
    print("current-current correlation for each flux")
    print("columns: flux_index phi1/2pi phi2/2pi q1 q2 JJxx JJxy JJyx JJyy")
    for flux_id, flux in enumerate(flux_grid):
        for q_id, q in enumerate(klist):
            jj = JJ_by_flux[flux_id, q_id]
            print(f"{flux_id:4d} {flux[0]: .8f} {flux[1]: .8f} {q[0]:4d} {q[1]:4d} {jj[0]: .8e} {jj[1]: .8e} {jj[2]: .8e} {jj[3]: .8e}")

    print()
    print("twist-averaged current-current correlation")
    print("columns: q1 q2 JJxx JJxy JJyx JJyy")
    for q, jj in zip(klist, JJ_avg):
        print(f"{q[0]:4d} {q[1]:4d} {jj[0]: .8e} {jj[1]: .8e} {jj[2]: .8e} {jj[3]: .8e}")


def manybody_diagonalize(Len, group, up_configs, singleE, k1234, V1234):
    configs = up_configs[group]
    dim = len(configs)
    rows, cols, vals = get_matrix_elements(configs, k1234, np.array(V1234, dtype=np.complex128))
    rows.extend(range(dim))
    cols.extend(range(dim))
    vals.extend(singleE[group])
    matrix = sp.csc_matrix((vals, (rows, cols)), shape=(dim, dim))

    if Len > 1:
        eigs, vecs = eigsh(matrix, k=1, which="SA", return_eigenvectors=True)
    else:
        eigs, vecs = eigsh(matrix, k=3, which="SA", return_eigenvectors=True)
    print(eigs.real)
    return vecs, eigs.real


def get_matrix_elements(configs, k1234, V1234):
    config_lookup = {value: index for index, value in enumerate(configs)}
    vals = []
    row = []
    col = []
    for ks, V in zip(k1234, V1234):
        k1, k2, k3, k4 = ks[0], ks[1], ks[2], ks[3]
        config_k3 = getspin(configs, k3)
        valid_k3 = (config_k3 == 1).nonzero()[0]
        f3, new_config = c_an(configs[valid_k3], k3)
        valid_k4 = (getspin(new_config, k4) == 1).nonzero()[0]
        f4, new_config = c_an(new_config[valid_k4], k4)
        f34 = f3[valid_k4] * f4

        valid_k2 = (getspin(new_config, k2) == 0).nonzero()[0]
        f2, new_config = c_dag(new_config[valid_k2], k2)
        f2 = f34[valid_k2] * f2

        valid_k1 = (getspin(new_config, k1) == 0).nonzero()[0]
        f1, new_config = c_dag(new_config[valid_k1], k1)
        fsign = f2[valid_k1] * f1

        old_ids = valid_k3[valid_k4[valid_k2[valid_k1]]]
        new_ids = np.array([config_lookup[nc] for nc in new_config]).astype(int)
        col.extend(old_ids)
        row.extend(new_ids)
        vals.extend(fsign * V * 0.5)
    return row, col, vals


def groupingKsum(configs, klist, single_E, N1, N2):
    binary = np.fliplr(config_array(configs, N1 * N2))
    occ_ksum = binary @ klist
    _, occ_ksum = compute_reduce_k(np.array(occ_ksum), N1, N2)
    Ktot, groups = two_index_unique(occ_ksum, N2)
    single_manybody_E = binary @ single_E.flatten(order="F")
    return Ktot, groups, single_manybody_E


def sample_k_q(N1, N2, Nmax):
    k1list = np.arange(N1, dtype=np.int32)
    k2list = np.arange(N2, dtype=np.int32)
    k12list = np.array([[k1list[i], k2list[j]] for j in range(N2) for i in range(N1)], dtype=np.int32)

    glist = np.arange(-Nmax, Nmax + 1, dtype=np.int32)
    glist = np.array([[glist[i], glist[j]] for j in range(2 * Nmax + 1) for i in range(2 * Nmax + 1)], dtype=np.int32)
    glist[:, 0] *= N1
    glist[:, 1] *= N2
    qlist = np.tile(k12list, [(2 * Nmax + 1) ** 2, 1]) + np.repeat(glist, N1 * N2, axis=0)
    return k12list, qlist


@njit
def compute_reduce_k(k12, N1, N2):
    gx, x_reduced = np.divmod(k12, np.array([N1, N2]))
    return gx, x_reduced


@njit
def convert_to_1_id(k1k2list, N):
    return k1k2list[:, 0] * N + k1k2list[:, 1]


@njit
def convert_to_2_id(klist, N):
    x, y = np.divmod(klist, N)
    return np.vstack((x, y)).T


@njit
def two_index_unique(testarray, N):
    array_1d = convert_to_1_id(testarray, N)
    unique_id = np.unique(array_1d)
    unique_indices = List()
    for uid in unique_id:
        unique_indices.append((array_1d == uid).nonzero()[0])
    return convert_to_2_id(unique_id, N), unique_indices


def eigh_sorted(A):
    vals, vecs = np.linalg.eigh(A)
    idx = vals.argsort()[::-1]
    return vals[idx], vecs[:, idx]


def g_vecs(theta, a0):
    g1 = np.array([4 * pi * theta / (sqrt(3) * a0), 0])
    return np.asarray([rot_mat(j * pi / 3) @ g1 for j in range(6)])


def reciprocal_vecs(gvecs):
    return gvecs[0], gvecs[2]


def rot_mat(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def config_array(configs, Nsys):
    strings = list(map(lambda x: format(x, "0" + str(Nsys) + "b"), configs))
    return np.array([[int(bit) for bit in binary] for binary in strings])


def bitstring_config(sys_size, num_particle):
    configs = []
    for ones_indices in combinations(range(sys_size), num_particle):
        bits = ["0"] * sys_size
        for idx in ones_indices:
            bits[idx] = "1"
        configs.append(int("".join(bits), 2))
    return configs


def getspin(b, i):
    return (b >> i) & 1


def bitflip(b, i):
    return b ^ (1 << i)


def lcounter(b, i):
    num = b >> (i + 1)
    return np.array([bin(n).count("1") for n in num])


def c_an(configs, site):
    return 1 - 2 * (lcounter(configs, site) % 2), bitflip(configs, site)


def c_dag(configs, site):
    return 1 - 2 * (lcounter(configs, site) % 2), bitflip(configs, site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal twist-averaged JJ(q) and spectral-flow calculation.")
    parser.add_argument("--theta", type=float, default=2.0)
    parser.add_argument("--n1", type=int, default=3)
    parser.add_argument("--n2", type=int, default=3)
    parser.add_argument("--nup", type=int, default=3)
    parser.add_argument("--vd", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=14)
    parser.add_argument("--gs-ktot-ind", type=int, nargs="+", default=[0])
    parser.add_argument("--output", default="JJcorr_TBC_minimal.npz")
    parser.add_argument("--n-jobs", type=int, default=0, help="Number of parallel flux workers. Use 0 for all cores.")
    args = parser.parse_args()

    main(args.theta, args.n1, args.n2, args.nup, args.vd, args.ntheta, args.gs_ktot_ind, args.output, args.n_jobs)
