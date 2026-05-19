#!/usr/bin/env python3

import sys
import time
import csv
import traceback
from pathlib import Path
from collections import defaultdict

import yaml
import torch
sys.path.insert(0, "src")
from reconstruction import load_trained_model, reconstruct_from_center

N_RUNS = 1

def create_polystyrene_chain(n_repeat_units: int) -> tuple:
    """
    Create a polystyrene chain.

    Repeat unit: -CH2-CH(C6H5)-
    Each unit contains:
      - 1 methylene carbon (CH2): 1C + 2H
      - 1 methine carbon (CH):   1C + 1H
      - 1 phenyl ring (C6H5):    6C + 5H
    Total: 8C + 8H = 16 atoms per repeat unit

    Args:
        n_repeat_units: Number of -CH2-CH(Ph)- repeat units

    Returns:
        elements: List of element symbols
        bonds:    List of bond dicts with 'i', 'j', 'order' keys
    """
    elements = []
    bonds = []
    atom_idx = 0
    prev_methine = None  # index of the CH carbon from the previous unit

    for unit in range(n_repeat_units):
        # ── methylene: -CH2- ──────────────────────────────────────────────────
        c_methylene = atom_idx;  elements.append('C');  atom_idx += 1
        h1 = atom_idx;           elements.append('H');  atom_idx += 1
        h2 = atom_idx;           elements.append('H');  atom_idx += 1

        bonds.append({'i': c_methylene, 'j': h1, 'order': 1.0})
        bonds.append({'i': c_methylene, 'j': h2, 'order': 1.0})

        # backbone bond: previous CH → this CH2
        if prev_methine is not None:
            bonds.append({'i': prev_methine, 'j': c_methylene, 'order': 1.0})

        # ── methine: -CH(Ph)- ─────────────────────────────────────────────────
        c_methine = atom_idx;  elements.append('C');  atom_idx += 1
        h3 = atom_idx;         elements.append('H');  atom_idx += 1

        bonds.append({'i': c_methine, 'j': h3,         'order': 1.0})
        bonds.append({'i': c_methylene, 'j': c_methine, 'order': 1.0})

        # ── phenyl ring: C6H5 ─────────────────────────────────────────────────
        # ph[0] = ipso (bonded to c_methine, no H)
        # ph[1..5] = ortho, meta, para, meta, ortho (each carries one H)
        ph = []
        for _ in range(6):
            c = atom_idx;  elements.append('C');  atom_idx += 1
            ph.append(c)

        # aromatic C-C ring bonds
        for k in range(6):
            bonds.append({'i': ph[k], 'j': ph[(k + 1) % 6], 'order': 1.5})

        # ipso bond to backbone methine
        bonds.append({'i': c_methine, 'j': ph[0], 'order': 1.0})

        # H on ortho/meta/para positions (all except ipso)
        for k in range(1, 6):
            h = atom_idx;  elements.append('H');  atom_idx += 1
            bonds.append({'i': ph[k], 'j': h, 'order': 1.0})

        prev_methine = c_methine

    # ── terminal H caps ───────────────────────────────────────────────────────
    # head: extra H on the first CH2 (atom 0)
    h_head = atom_idx;  elements.append('H');  atom_idx += 1
    bonds.append({'i': 0, 'j': h_head, 'order': 1.0})

    # tail: extra H on the last CH
    h_tail = atom_idx;  elements.append('H');  atom_idx += 1
    bonds.append({'i': prev_methine, 'j': h_tail, 'order': 1.0})

    print("Created polystyrene chain:")
    print(f"  Repeat units : {n_repeat_units}")
    print(f"  Total atoms  : {len(elements)}")
    print(f"  Total bonds  : {len(bonds)}")
    print(f"  Repeat unit  : -CH2-CH(C6H5)-")

    return elements, bonds

def create_polyester_chain(n_repeat_units: int) -> tuple:
    """
    Create a simplified aliphatic polyester chain.

    Repeat unit:
        -O-CH2-CH2-C(=O)-

    Args:
        n_repeat_units: Number of polyester repeat units

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []

    atom_idx = 0
    previous_carbonyl_c = None

    for unit in range(n_repeat_units):
        # Ether oxygen
        o_ether = atom_idx
        elements.append("O")
        atom_idx += 1

        # CH2 carbon 1
        c1 = atom_idx
        elements.append("C")
        atom_idx += 1

        # CH2 carbon 2
        c2 = atom_idx
        elements.append("C")
        atom_idx += 1

        # Carbonyl carbon
        c_carbonyl = atom_idx
        elements.append("C")
        atom_idx += 1

        # Carbonyl oxygen
        o_carbonyl = atom_idx
        elements.append("O")
        atom_idx += 1

        # Connect previous repeat unit to current ester oxygen
        if previous_carbonyl_c is not None:
            bonds.append({'i': previous_carbonyl_c, 'j': o_ether, 'order': 1.0})

        # Main polyester backbone: O-CH2-CH2-C(=O)
        bonds.append({'i': o_ether, 'j': c1, 'order': 1.0})
        bonds.append({'i': c1, 'j': c2, 'order': 1.0})
        bonds.append({'i': c2, 'j': c_carbonyl, 'order': 1.0})
        bonds.append({'i': c_carbonyl, 'j': o_carbonyl, 'order': 2.0})

        # Hydrogens on CH2 groups
        for carbon in [c1, c2]:
            for _ in range(2):
                h = atom_idx
                elements.append("H")
                atom_idx += 1
                bonds.append({'i': carbon, 'j': h, 'order': 1.0})

        previous_carbonyl_c = c_carbonyl

    # Terminal hydrogens: H-O-...-C(=O)-H simplified
    h_start = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({'i': 0, 'j': h_start, 'order': 1.0})

    h_end = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({'i': previous_carbonyl_c, 'j': h_end, 'order': 1.0})

    return elements, bonds

def create_polytetrafluoroethylene_chain(n_carbons: int) -> tuple:
    """
    Create a simple polytetrafluoroethylene chain with n carbon atoms.

    Args:
        n_carbons: Number of carbon atoms in the chain

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []

    atom_idx = 0
    for c_idx in range(n_carbons):
        # Add carbon
        c_atom_idx = atom_idx
        elements.append('C')
        atom_idx += 1

        # Add 2 hydrogens per carbon
        h1_idx = atom_idx
        elements.append('F')
        atom_idx += 1

        h2_idx = atom_idx
        elements.append('F')
        atom_idx += 1

        # C-H bonds
        bonds.append({'i': c_atom_idx, 'j': h1_idx, 'order': 1.0})
        bonds.append({'i': c_atom_idx, 'j': h2_idx, 'order': 1.0})

        # C-C backbone bond (except for last carbon)
        if c_idx < n_carbons - 1:
            next_c_idx = c_atom_idx + 3  # Next carbon is 3 atoms away
            bonds.append({'i': c_atom_idx, 'j': next_c_idx, 'order': 1.0})

    return elements, bonds

def create_peg_chain(n_repeat_units: int) -> tuple:
    """
    Create a simplified PEG / polyethylene glycol chain.

    Repeat unit:
        -CH2-CH2-O-

    Args:
        n_repeat_units: Number of PEG repeat units

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []

    atom_idx = 0
    previous_o = None

    for unit in range(n_repeat_units):
        # CH2 carbon 1
        c1 = atom_idx
        elements.append("C")
        atom_idx += 1

        # CH2 carbon 2
        c2 = atom_idx
        elements.append("C")
        atom_idx += 1

        # Ether oxygen
        o = atom_idx
        elements.append("O")
        atom_idx += 1

        # Connect previous repeat unit to current unit
        if previous_o is not None:
            bonds.append({'i': previous_o, 'j': c1, 'order': 1.0})

        # Main PEG backbone: C-C-O
        bonds.append({'i': c1, 'j': c2, 'order': 1.0})
        bonds.append({'i': c2, 'j': o, 'order': 1.0})

        # Hydrogens on each CH2
        for carbon in [c1, c2]:
            for _ in range(2):
                h = atom_idx
                elements.append("H")
                atom_idx += 1
                bonds.append({'i': carbon, 'j': h, 'order': 1.0})

        previous_o = o

    # Optional terminal hydrogens: HO-(CH2-CH2-O)n-H
    h_start = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({'i': 0, 'j': h_start, 'order': 1.0})

    h_end = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({'i': previous_o, 'j': h_end, 'order': 1.0})

    return elements, bonds


# =====================================================
# OUTPUT FOLDER
# =====================================================
OUT_DIR = Path("reconstruction_time")
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV = OUT_DIR / "polymer_reconstruction_time_benchmark.csv"

# =====================================================
# SIZES
# =====================================================
SIZES = (
    list(range(1000, 10001, 1000))
    + list(range(20000, 100001, 10000))
    + list(range(200000, 900001, 100000))
)

# =====================================================
# POLYMERS
# =====================================================
POLYMERS = {
    "polystyrene": create_polystyrene_chain,
    "ptfe": create_polytetrafluoroethylene_chain,
    "peg": create_peg_chain,
    "polyester": create_polyester_chain,
}


# =====================================================
# HELPERS
# =====================================================
def choose_start_atom(elements, bonds):
    adjacency = defaultdict(list)

    for bond in bonds:
        i, j = bond["i"], bond["j"]
        adjacency[i].append(j)
        adjacency[j].append(i)

    candidates = [i for i, e in enumerate(elements) if e != "H"]
    if not candidates:
        candidates = list(range(len(elements)))

    return max(candidates, key=lambda x: len(adjacency[x]))


def save_xyz(elements, positions, filepath, comment="Generated Polymer"):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{len(elements)}\n")
        f.write(comment + "\n")

        for elem, pos in zip(elements, positions):
            f.write(
                f"{elem:2s} "
                f"{pos[0]:14.6f} "
                f"{pos[1]:14.6f} "
                f"{pos[2]:14.6f}\n"
            )


def benchmark_one(polymer_name, builder, size, model, element_vocab, device, max_bonds):
    t0 = time.perf_counter()

    elements, bonds = builder(size)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    start_atom = choose_start_atom(elements, bonds)

    with torch.no_grad():
        positions, _, _, _, _ = reconstruct_from_center(
            model=model,
            elements=elements,
            bonds_data=bonds,
            element_to_idx=element_vocab,
            device=device,
            max_bonds=max_bonds,
            start_atom=start_atom,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    # Save coordinates
    xyz_file = OUT_DIR / f"{polymer_name}_{size}.xyz"

    save_xyz(
        elements,
        positions,
        xyz_file,
        comment=f"{polymer_name} size={size}"
    )

    return {
        "polymer": polymer_name,
        "size_input": size,
        "n_atoms": len(elements),
        "n_bonds": len(bonds),
        "generation_time_s": t1 - t0,
        "reconstruction_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "xyz_file": xyz_file.name,
        "status": "ok",
        "error": "",
    }


def append_result(row):
    fieldnames = [
        "polymer",
        "size_input",
        "run_idx",
        "n_runs",
        "n_atoms",
        "n_bonds",
        "generation_time_s",
        "reconstruction_time_s",
        "total_time_s",
        "xyz_file",
        "status",
        "error",
    ]

    write_header = not OUT_CSV.exists()

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 70)
    print("Polymer Reconstruction Time Benchmark")
    print("=" * 70)

    print("\n[1] Loading model...")

    model, element_vocab, device = load_trained_model(
        checkpoint_path="checkpoints/best_model.pt",
        config_path="config/config.yaml",
    )

    model.eval()

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    max_bonds = config["data"]["max_bonds"]

    # =====================================================
    # WARM-UP
    # =====================================================
    print("\n[Warm-up] Running one unlogged reconstruction...")

    warm_elements, warm_bonds = create_polystyrene_chain(2)
    warm_start_atom = choose_start_atom(warm_elements, warm_bonds)

    with torch.no_grad():
        _ = reconstruct_from_center(
            model=model,
            elements=warm_elements,
            bonds_data=warm_bonds,
            element_to_idx=element_vocab,
            device=device,
            max_bonds=max_bonds,
            start_atom=warm_start_atom,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("[Warm-up] Done.")

    print(f"\nOutput folder: {OUT_DIR.resolve()}")
    print(f"Benchmark log: {OUT_CSV.name}")
    print(f"Number of benchmark repeats: {N_RUNS}")

    print("\n[2] Starting benchmark...\n")

    try:
        for size in SIZES:

            print("\n" + "=" * 70)
            print(f"MONOMER / REPEAT UNIT QTY: {size}")
            print("=" * 70)

            for polymer_name, builder in POLYMERS.items():

                print("\n" + "-" * 70)
                print(f"{polymer_name} | size={size}")
                print("-" * 70)

                for run_idx in range(1, N_RUNS + 1):

                    print(
                        f"{polymer_name:12s} "
                        f"size={size:7d} "
                        f"run={run_idx:3d}/{N_RUNS}",
                        end=" | ",
                        flush=True,
                    )

                    try:
                        row = benchmark_one(
                            polymer_name=polymer_name,
                            builder=builder,
                            size=size,
                            model=model,
                            element_vocab=element_vocab,
                            device=device,
                            max_bonds=max_bonds,
                        )

                        row["run_idx"] = run_idx
                        row["n_runs"] = N_RUNS

                        append_result(row)

                        print(
                            f"atoms={row['n_atoms']:9d} "
                            f"bonds={row['n_bonds']:9d} "
                            f"recon={row['reconstruction_time_s']:.4f}s "
                            f"saved={row['xyz_file']}"
                        )

                    except Exception as e:
                        row = {
                            "polymer": polymer_name,
                            "size_input": size,
                            "n_atoms": None,
                            "n_bonds": None,
                            "generation_time_s": None,
                            "reconstruction_time_s": None,
                            "total_time_s": None,
                            "xyz_file": "",
                            "status": "failed",
                            "error": repr(e),
                            "run_idx": run_idx,
                            "n_runs": N_RUNS,
                        }

                        append_result(row)

                        print(f"FAILED: {e}")
                        traceback.print_exc()
                        break

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print("Previous completed benchmark rows are already saved.")
        print(f"Saved in: {OUT_CSV.resolve()}")
        return

    print("\nBenchmark complete.")
    print(f"All results saved in: {OUT_DIR.resolve()}")



if __name__ == "__main__":
    main()