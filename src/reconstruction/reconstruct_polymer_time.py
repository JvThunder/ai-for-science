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
from reconstruction.reconstruction import load_trained_model, reconstruct_from_center
from reconstruction.molecules import Polystyrene, PTFE, PEG, Polyester

N_RUNS = 1

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
    "polystyrene": Polystyrene.create_topology,
    "ptfe":        PTFE.create_topology,
    "peg":         PEG.create_topology,
    "polyester":   Polyester.create_topology,
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

    warm_elements, warm_bonds = Polystyrene.create_topology(2)
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