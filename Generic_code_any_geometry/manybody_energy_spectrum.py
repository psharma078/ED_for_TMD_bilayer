import argparse
import multiprocessing
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy import cos, exp, pi, sin, sqrt
from scipy.sparse.linalg import eigsh


num_cores = multiprocessing.cpu_count()
epsilon_r = 10.0


def main(
    theta_deg,
    nup,
    vd,
    l1=(6, 0),
    l2=(-2, 4),
    nk=8,
    nev=4,
    n_jobs=None,
    output_prefix=None,
):
    if n_jobs is None or n_jobs <= 0:
        n_jobs = num_cores

    theta = theta_deg / 180.0 * pi
    a0 = 3.52e-10
    aM = a0 / (2 * np.sin(theta / 2))
    gvecs = g_vecs(theta, aM)
    Q1, Q2 = reciprocal_vecs(gvecs)
    kappa_p = (2 * Q1 + Q2) / 3
    kappa_m = (Q1 - Q2) / 3

    T1, T2, fold_vectors, klabels = tilted_geometry(aM, l1, l2)
    n_k = len(klabels)
    if nup < 0 or nup > n_k:
        raise ValueError(f"nup must be between 0 and Norb={n_k}; got {nup}.")
    k_lookup = {tuple(k): i for i, k in enumerate(klabels)}

    n_g = (2 * nk + 1) ** 2
    nk_vals = np.arange(-nk, nk + 1)
    ones = np.ones(2 * nk + 1)
    mmlist = np.kron(nk_vals, ones)
    nnlist = np.kron(ones, nk_vals)
    Qxlist = nnlist * Q1[0] + mmlist * Q2[0]
    Qylist = nnlist * Q1[1] + mmlist * Q2[1]

    diagm = np.diag(np.ones(2 * nk, dtype=np.complex128), -1)
    diagp = diagm.T
    diag_Nk = np.eye(2 * nk + 1, dtype=np.complex128)
    delta_nn_mmp1 = np.kron(diagm, diag_Nk)
    delta_nnp1_mm = np.kron(diag_Nk, diagm)
    delta_nnm1_mmm1 = np.kron(diagp, diagp)
    delta_nn_mm = np.kron(diag_Nk, diag_Nk)
    delta_nn_mmm1 = delta_nn_mmp1.T

    halfdim = n_g
    ham_dim = 2 * halfdim

    def gen_layer_hamiltonian(k):
        me = 9.10938356e-31
        mass = 0.62 * me
        hbar = 1.054571817e-34
        J_to_meV = 6.24150636e21
        prefactor = hbar**2 / (2 * mass) * J_to_meV

        kx_top = Qxlist + k[0] - kappa_p[0]
        ky_top = Qylist + k[1] - kappa_p[1]
        kx_bottom = Qxlist + k[0] - kappa_m[0]
        ky_bottom = Qylist + k[1] - kappa_m[1]

        psi = -91.0 / 180.0 * pi
        V = 11.2
        H_top = -V * (delta_nn_mmp1 + delta_nnp1_mm + delta_nnm1_mmm1) * exp(1j * psi)
        H_top += H_top.conj().T
        H_bottom = H_top.conj()

        np.fill_diagonal(H_top, (kx_top**2 + ky_top**2) * prefactor + vd / 2.0)
        np.fill_diagonal(H_bottom, (kx_bottom**2 + ky_bottom**2) * prefactor - vd / 2.0)
        return H_top, H_bottom

    def construct_ham(k):
        H_top, H_bottom = gen_layer_hamiltonian(k)
        w = 13.3
        Delta_T = -w * (delta_nn_mm + delta_nnm1_mmm1 + delta_nn_mmm1)
        ham = np.zeros((ham_dim, ham_dim), dtype=np.complex128)
        ham[:halfdim, :halfdim] = H_top
        ham[halfdim:, halfdim:] = H_bottom
        ham[:halfdim, halfdim:] = Delta_T.conj().T
        ham[halfdim:, :halfdim] = Delta_T
        return ham

    reducer = MomentumReducer(fold_vectors, klabels)
    glabels = sample_g_labels(nk)
    qlabels = np.vstack([k + g @ fold_vectors for g in glabels for k in klabels]).astype(np.int64)
    plus_kid, plus_gid = precompute_shift_table(klabels, qlabels, reducer, k_lookup, nk)
    minus_kid, minus_gid = precompute_shift_table(klabels, -qlabels, reducer, k_lookup, nk)

    print("L1 =", tuple(l1), "L2 =", tuple(l2), "Norb =", n_k)
    print("representatives:")
    for i, k in enumerate(klabels):
        print(i, k)

    print("computing single-particle band...")
    kxylist = klabels @ np.array([T1, T2])
    result = Parallel(n_jobs=n_jobs)(delayed(eigh_sorted)(construct_ham(k)) for k in kxylist)
    single_all = np.array([r[0] for r in result])
    singleE_orb = single_all[:, -1]
    for k, e in zip(klabels, singleE_orb):
        print(k, e)

    eigvecs = np.array([r[1][:, -1] for r in result], dtype=np.complex128)
    eigvecs = eigvecs.reshape(n_k, 2 * nk + 1, 2 * nk + 1, 2, order="F")

    print("building form factors...")
    Fmat = build_form_factors(eigvecs, nk)

    print("building Coulomb table...")
    Vlist = coulomb_potential(qlabels, T1, T2, aM, n_k)
    k1234, V1234 = coulomb_matrix_elements(
        plus_kid, plus_gid, minus_kid, minus_gid, Fmat, Vlist, n_k, n_g
    )
    print("Coulomb terms:", len(V1234))

    up_configs = np.array(bitstring_config(n_k, nup), dtype=np.int64)
    binary_lists = np.fliplr(config_array(up_configs, n_k))
    Ktot, groups, single_manybody_E = group_configs(binary_lists, klabels, reducer, singleE_orb)

    print("diagonalizing K sectors...")
    solver = lambda gid: manybody_energies(gid, up_configs, single_manybody_E, k1234, V1234, nev)
    spectrum = Parallel(n_jobs=n_jobs)(delayed(solver)(gid) for gid in groups)

    print("ktot,         Eigen values (meV)")
    for K, eigs in zip(Ktot, spectrum):
        print(K, eigs)
    print()
    print("Eigenvalues sorted")
    print(np.sort(np.concatenate(spectrum)))

    if output_prefix:
        momentum_ids = momentum_1d_ids(Ktot, k_lookup)
        plot_spectrum(Ktot, momentum_ids, spectrum, output_prefix + ".png")
        np.savez(
            output_prefix + ".npz",
            Ktot=Ktot,
            momentum_ids=momentum_ids,
            spectrum=np.array(spectrum, dtype=object),
            klabels=klabels,
            l1=np.array(l1),
            l2=np.array(l2),
            theta_deg=theta_deg,
            nup=nup,
            vd=vd,
            nk=nk,
        )
        print("saved", output_prefix + ".png")
        print("saved", output_prefix + ".npz")

    return Ktot, spectrum


class MomentumReducer:
    def __init__(self, fold_vectors, reps):
        self.fold_vectors = np.asarray(fold_vectors, dtype=np.int64)
        self.reps = np.asarray(reps, dtype=np.int64)
        self.det = int(round(abs(np.linalg.det(self.fold_vectors))))
        self.rep_by_key = {self.coset_key(rep): rep.copy() for rep in self.reps}
        if len(self.rep_by_key) != self.det:
            raise ValueError("Momentum representatives do not contain exactly one label per coset.")

    def coset_key(self, label):
        label = np.asarray(label, dtype=np.int64)
        a, b = self.fold_vectors[0]
        c, d = self.fold_vectors[1]
        det = a * d - b * c
        nums = np.array([label[0] * d - label[1] * c, -label[0] * b + label[1] * a], dtype=np.int64)
        return tuple(np.mod(nums, abs(det)))

    def reduce(self, label):
        label = np.asarray(label, dtype=np.int64)
        rep = self.rep_by_key[self.coset_key(label)]
        diff = label - rep
        coeff = solve_integer_row(diff, self.fold_vectors)
        return coeff, rep


def solve_integer_row(diff, fold_vectors):
    a, b = fold_vectors[0]
    c, d = fold_vectors[1]
    det = a * d - b * c
    x, y = int(diff[0]), int(diff[1])
    n0 = x * d - y * c
    n1 = -x * b + y * a
    if n0 % det != 0 or n1 % det != 0:
        raise ValueError(f"{diff} is not in the folding lattice.")
    return np.array([n0 // det, n1 // det], dtype=np.int64)


def tilted_geometry(aM, l1, l2):
    l1 = np.asarray(l1, dtype=np.int64)
    l2 = np.asarray(l2, dtype=np.int64)
    a1 = aM * np.array([sqrt(3) / 2, 0.5])
    a2 = aM * np.array([0.0, 1.0])
    L1 = l1[0] * a1 + l1[1] * a2
    L2 = l2[0] * a1 + l2[1] * a2
    area = cross2d(L1, L2)
    if abs(area) < 1e-30:
        raise ValueError(f"L1={tuple(l1)} and L2={tuple(l2)} are linearly dependent.")
    T1 = 2 * pi * np.array([L2[1], -L2[0]]) / area
    T2 = 2 * pi * np.array([-L1[1], L1[0]]) / area

    fold_vectors = np.array([[l1[0], l2[0]], [l1[1], l2[1]]], dtype=np.int64)
    if int(round(abs(np.linalg.det(fold_vectors)))) == 0:
        raise ValueError(f"L1={tuple(l1)} and L2={tuple(l2)} give zero momentum-cell area.")
    reps = positive_representatives(fold_vectors)
    return T1, T2, fold_vectors, reps


def positive_representatives(fold_vectors):
    """Return one nonnegative integer representative for every momentum coset."""
    rect_reps = rectangular_positive_representatives(fold_vectors)
    if rect_reps is not None:
        return rect_reps

    det = int(round(abs(np.linalg.det(fold_vectors))))
    limit = max(1, int(np.max(np.abs(fold_vectors))))
    while True:
        candidates = np.array(
            [[i, j] for j in range(limit + 1) for i in range(limit + 1)],
            dtype=np.int64,
        )
        order = np.lexsort((candidates[:, 1], candidates[:, 0], np.sum(candidates * candidates, axis=1)))
        reps = []
        keys = set()
        reducer_stub = MomentumReducer.__new__(MomentumReducer)
        reducer_stub.fold_vectors = np.asarray(fold_vectors, dtype=np.int64)
        reducer_stub.det = det
        for cand in candidates[order]:
            key = MomentumReducer.coset_key(reducer_stub, cand)
            if key not in keys:
                keys.add(key)
                reps.append(cand.copy())
                if len(reps) == det:
                    return np.array(reps, dtype=np.int64)
        limit *= 2


def rectangular_positive_representatives(fold_vectors):
    det = int(round(abs(np.linalg.det(fold_vectors))))
    reducer_stub = MomentumReducer.__new__(MomentumReducer)
    reducer_stub.fold_vectors = np.asarray(fold_vectors, dtype=np.int64)
    reducer_stub.det = det

    valid_shapes = []
    for n1 in range(1, det + 1):
        if det % n1 != 0:
            continue
        n2 = det // n1
        if n1 < n2:
            continue
        reps = np.array([[i, j] for j in range(n2) for i in range(n1)], dtype=np.int64)
        keys = {MomentumReducer.coset_key(reducer_stub, rep) for rep in reps}
        if len(keys) == det:
            valid_shapes.append((n1 / n2, n1, n2, reps))

    if not valid_shapes:
        return None
    _, _, _, reps = min(valid_shapes, key=lambda item: item[0])
    return reps


def sample_g_labels(nk):
    vals = np.arange(-nk, nk + 1, dtype=np.int64)
    return np.array([[i, j] for j in vals for i in vals], dtype=np.int64)


def precompute_shift_table(klabels, shifts, reducer, k_lookup, nk):
    n_k = len(klabels)
    n_shift = len(shifts)
    kid = np.empty((n_k, n_shift), dtype=np.int64)
    gid = np.empty((n_k, n_shift), dtype=np.int64)
    width = 2 * nk + 1
    for i, k in enumerate(klabels):
        for j, q in enumerate(shifts):
            g, rep = reducer.reduce(k + q)
            kid[i, j] = k_lookup[tuple(rep)]
            gs = g + nk
            if np.any(gs < 0) or np.any(gs >= width):
                gid[i, j] = -1
            else:
                gid[i, j] = gs[0] + width * gs[1]
    return kid, gid


def overlap_slices(shift, size):
    if shift >= 0:
        return shift, size, 0, size - shift
    return 0, size + shift, -shift, size


def build_form_factors(eigvecs, nk):
    n_k = eigvecs.shape[0]
    width = 2 * nk + 1
    n_g = width * width
    Fmat = np.zeros((n_g, n_k, n_k), dtype=np.complex128)
    for gx in range(-nk, nk + 1):
        x1a, x1b, x2a, x2b = overlap_slices(gx, width)
        for gy in range(-nk, nk + 1):
            y1a, y1b, y2a, y2b = overlap_slices(gy, width)
            gid = (gx + nk) + width * (gy + nk)
            left = eigvecs[:, x1a:x1b, y1a:y1b, :].reshape(n_k, -1)
            right = eigvecs[:, x2a:x2b, y2a:y2b, :].reshape(n_k, -1)
            # The original code flattens Fmat_up with order="F" but later indexes
            # it as g*n_k**2 + k_old*n_k + k_new. That retrieves
            # Fmat_up[k_new, k_old, g], so store the first two indices
            # transposed here to preserve the old matrix-element convention.
            Fmat[gid] = (left.conj() @ right.T).T
    return Fmat.reshape(-1, order="C")


def coulomb_potential(qlabels, T1, T2, aM, n_k):
    qxy = qlabels @ np.array([T1, T2])
    absq = np.linalg.norm(qxy, axis=1)
    k0 = 8.99e9
    J_to_meV = 6.242e21
    e_charge = 1.602e-19
    area = np.sqrt(3) / 2 * n_k * aM**2
    Vc = np.zeros_like(absq)
    nonzero = absq > 1e-14
    Vc[nonzero] = 2 * np.pi * e_charge**2 / (epsilon_r * absq[nonzero]) * J_to_meV / area * k0
    return Vc


def compute_coulomb(k3_id, k4_id, plus_kid, plus_gid, minus_kid, minus_gid, Fmat, Vlist, n_k, n_g):
    k1_id = plus_kid[k3_id]
    g1_id = plus_gid[k3_id]
    k2_id = minus_kid[k4_id]
    g2_id = minus_gid[k4_id]
    keep = (g1_id >= 0) & (g1_id < n_g) & (g2_id >= 0) & (g2_id < n_g) & (k1_id < k2_id)
    if not np.any(keep):
        return {}, np.empty((0, 2), dtype=np.int64)

    k1 = k1_id[keep]
    k2 = k2_id[keep]
    g1 = g1_id[keep]
    g2 = g2_id[keep]
    values = Fmat[g1 * n_k * n_k + k3_id * n_k + k1]
    values *= Fmat[g2 * n_k * n_k + k4_id * n_k + k2] * Vlist[keep]

    terms = {}
    for a, b, val in zip(k1, k2, values):
        key = (int(a), int(b))
        terms[key] = terms.get(key, 0.0) + val
    pairs = np.array(sorted(terms), dtype=np.int64)
    return terms, pairs


def coulomb_matrix_elements(plus_kid, plus_gid, minus_kid, minus_gid, Fmat, Vlist, n_k, n_g):
    k1234 = []
    V1234 = []
    for ii in range(n_k):
        for jj in range(ii):
            exchange, _ = compute_coulomb(ii, jj, plus_kid, plus_gid, minus_kid, minus_gid, Fmat, Vlist, n_k, n_g)
            direct, pairs = compute_coulomb(jj, ii, plus_kid, plus_gid, minus_kid, minus_gid, Fmat, Vlist, n_k, n_g)
            for k1, k2 in pairs:
                key = (int(k1), int(k2))
                k1234.append([k1, k2, jj, ii])
                V1234.append(2 * (direct[key] - exchange.get(key, 0.0)))
    return np.array(k1234, dtype=np.int64), np.array(V1234, dtype=np.complex128)


def group_configs(binary_lists, klabels, reducer, singleE_orb):
    occ_ksum = binary_lists @ klabels
    occ_ksum = np.array([reducer.reduce(total)[1] for total in occ_ksum], dtype=np.int64)
    rep_to_id = {tuple(rep): i for i, rep in enumerate(klabels)}
    group_ids = np.array([rep_to_id[tuple(rep)] for rep in occ_ksum], dtype=np.int64)
    Ktot = []
    groups = []
    for uid in range(len(klabels)):
        idx = np.nonzero(group_ids == uid)[0]
        if len(idx) == 0:
            continue
        Ktot.append(klabels[uid].copy())
        groups.append(idx.astype(np.int64))
    single_manybody_E = binary_lists @ singleE_orb
    return np.array(Ktot, dtype=np.int64), groups, single_manybody_E


def get_matrixEle(configs, k1234, V1234):
    config_lookup = {value: index for index, value in enumerate(configs)}
    mat_ele = []
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
            mat_ele.append(fsign * V * 0.5)
            col.append(jj)
            row.append(config_lookup[newconfig])
    return row, col, mat_ele


def manybody_energies(configs_indx, up_configs, singleE, k1234, V1234, nev):
    dimHam = len(configs_indx)
    configs = up_configs[configs_indx]
    rows, cols, mat_ele = get_matrixEle(configs, k1234, V1234)
    rows.extend(range(dimHam))
    cols.extend(range(dimHam))
    mat_ele.extend(singleE[configs_indx])
    matrix = sp.csc_matrix((mat_ele, (rows, cols)), shape=(dimHam, dimHam))
    if dimHam <= nev + 1:
        return np.sort(np.linalg.eigvalsh(matrix.toarray()).real)[:nev]
    k = min(nev, dimHam - 2)
    eigs = eigsh(matrix, k=k, which="SA", sigma=None, return_eigenvectors=False)
    return np.sort(eigs.real)


def momentum_1d_ids(Ktot, k_lookup):
    return np.array([k_lookup[tuple(k)] for k in Ktot], dtype=np.int64)


def plot_spectrum(Ktot, momentum_ids, spectrum, filename):
    import matplotlib.pyplot as plt

    emin = min(np.min(eigs) for eigs in spectrum if len(eigs))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for x, eigs in zip(momentum_ids, spectrum):
        ax.scatter(np.full(len(eigs), x), np.asarray(eigs) - emin, color="black", s=16)
    ax.set_xlabel(r"$k_2 \times N_1 + k1$", fontsize=12)
    ax.set_ylabel(r"$E - E_{\min}$ (meV)", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def config_array(configs, Nsys):
    configs_bin = list(map(lambda x: format(x, "0" + str(Nsys) + "b"), configs))
    return np.array([[int(bit) for bit in binary] for binary in configs_bin])


def bitstring_config(sys_size, num_particle):
    decimal_numbers = []
    for ones_indices in combinations(range(sys_size), num_particle):
        binary_str = ["0"] * sys_size
        for idx in ones_indices:
            binary_str[idx] = "1"
        decimal_numbers.append(int("".join(binary_str), 2))
    return decimal_numbers


def getspin(b, i):
    return (b >> i) & 1


def bitflip(b, i):
    return b ^ (1 << i)


def spinless_fsigns(config, indices):
    i = min(indices[0], indices[1])
    j = max(indices[0], indices[1])
    k = min([indices[2], indices[3]])
    ll = max([indices[2], indices[3]])
    Fkl = np.array([getspin(config, x) for x in range(k + 1, ll)])
    Fkl = np.prod(1 - 2 * Fkl)
    new_config = bitflip(bitflip(config, k), ll)
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


def g_vecs(theta, aM):
    g1 = np.array([4 * pi / (sqrt(3) * aM), 0])
    return np.asarray([rot_mat(j * pi / 3) @ g1 for j in range(6)])


def reciprocal_vecs(gvecs):
    return gvecs[0], gvecs[2]


def rot_mat(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def eigh_sorted(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]
    return eigenValues[idx], eigenVectors[:, idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lowest many-body spectrum for a tilted moire cluster.")
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--nup", type=int, default=8)
    parser.add_argument("--vd", type=float, default=0.0)
    parser.add_argument("--l1", type=int, nargs=2, default=[6, 0], help="Real-space supercell vector L1 in a1,a2 coordinates.")
    parser.add_argument("--l2", type=int, nargs=2, default=[-2, 4], help="Real-space supercell vector L2 in a1,a2 coordinates.")
    parser.add_argument("--nk", type=int, default=8)
    parser.add_argument("--nev", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs. Use 0 for all cores.")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    main(
        args.theta,
        args.nup,
        args.vd,
        tuple(args.l1),
        tuple(args.l2),
        args.nk,
        args.nev,
        args.n_jobs,
        args.output_prefix,
    )
