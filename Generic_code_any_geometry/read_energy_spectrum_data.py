import argparse
import numpy as np


DEFAULT_FILE = "ED_tilted_v2_L1_6_0_L2_-2_4_nup_8_theta_2.0_nk_8.npz"


def main(filename, show_raw):
    data = np.load(filename, allow_pickle=True)

    print("file:", filename)
    print("arrays:")
    for key in data.files:
        arr = data[key]
        print(f"  {key:16s} shape={arr.shape} dtype={arr.dtype}")
    print()

    klabels = data["klabels"]
    Ktot = data["Ktot"]
    momentum_ids = data["momentum_ids"]
    spectrum = data["spectrum"]

    print("parameters")
    for key in ["l1", "l2", "nup", "nk", "theta_deg", "vd"]:
        if key in data.files:
            print(f"  {key} =", data[key])
    print()

    print("momentum representative table")
    print("columns: id k1 k2")
    for i, k in enumerate(klabels):
        print(f"{i:4d} {k[0]:4d} {k[1]:4d}")
    print()

    print("many-body spectrum by momentum sector")
    print("columns: sector_index momentum_id K1 K2 eigenvalues...")
    for sector_id, (mid, K, eigs) in enumerate(zip(momentum_ids, Ktot, spectrum)):
        eigs = np.asarray(eigs, dtype=float)
        eig_str = " ".join(f"{e: .10e}" for e in eigs)
        print(f"{sector_id:4d} {int(mid):4d} {K[0]:4d} {K[1]:4d} {eig_str}")
    print()

    all_eigs = np.sort(np.concatenate([np.asarray(e, dtype=float) for e in spectrum]))
    emin = all_eigs[0]
    print("sorted eigenvalues")
    print("columns: index E E-Emin")
    for i, e in enumerate(all_eigs):
        print(f"{i:4d} {e: .10e} {e - emin: .10e}")

    if show_raw:
        print()
        print("raw arrays")
        np.set_printoptions(threshold=np.inf, linewidth=200)
        for key in data.files:
            print(f"\n== {key} ==")
            print(data[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and print tilted v2 ED spectrum npz output.")
    parser.add_argument("filename", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("--show-raw", action="store_true", help="Print every raw array in full.")
    args = parser.parse_args()
    main(args.filename, args.show_raw)
