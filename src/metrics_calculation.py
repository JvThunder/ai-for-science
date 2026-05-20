#!/usr/bin/env python3

import sys
import math
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# src/ is the parent of this file; add it so utils.* imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model import LocalGeometryGCN
from utils.dataset import (
    parse_xyz_string,
    parse_pubchem_data,
    get_element_vocab,
    dfs_subgraph,
)
from utils.geometry import compute_local_geometry, get_neighbor_sort_keys
from utils.config import load_config

GEOMETRY_COLUMN = "DFT_PBE_USPP_Optimized_Geometry"
FLUSH_EVERY = 1000


# =====================================================
# BASIC HELPERS
# =====================================================
def pubchem_mapping_has_error(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return "error" in str(x).lower()


def load_trained_model(checkpoint_path, config_path, device="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)

    config = load_config(str(config_path))

    elements_txt = config["data"].get("elements_txt", "elements.txt")
    elements_txt = Path(config_path).parent / Path(elements_txt).name

    element_to_idx = get_element_vocab(txt_path=str(elements_txt))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # IMPORTANT: infer vocab size from checkpoint, not current elements.txt
    vocab_size = state_dict["element_embedding.weight"].shape[0]

    print(f"Checkpoint vocab size: {vocab_size}")
    print(f"Current element vocab size: {len(element_to_idx)}")

    model = LocalGeometryGCN(
        num_node_features=vocab_size,
        hidden_dim=config["model"]["hidden_dim"],
        max_bonds=config["data"]["max_bonds"],
        vocab_size=vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        depth=config["model"].get("depth", 1),
        use_bond_types=config["model"].get("use_bond_types", True),
        num_bond_types=config["model"].get("num_bond_types", 4),
        bond_embedding_dim=config["model"].get("bond_embedding_dim", 8),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, element_to_idx, device, config


def build_adjacency_and_orders(bonds, n_atoms):
    adjacency = defaultdict(list)
    bond_orders = {}
    bond_id_lookup = {}

    for bond_id, bond in enumerate(bonds):
        if "i" not in bond or "j" not in bond:
            continue

        i, j = bond["i"], bond["j"]

        if i >= n_atoms or j >= n_atoms or i == j:
            continue

        order = bond.get("order", np.nan)

        adjacency[i].append(j)
        adjacency[j].append(i)

        bond_orders[(i, j)] = order
        bond_orders[(j, i)] = order

        bond_id_lookup[(i, j)] = bond_id
        bond_id_lookup[(j, i)] = bond_id

    return adjacency, bond_orders, bond_id_lookup


def create_graph_data(
    coordinates,
    elements,
    bonds_data,
    element_to_idx,
    node_a,
    node_b,
    max_bonds,
    device,
    subgraph_depth=1,
):
    from torch_geometric.data import Data

    pos = torch.tensor(coordinates, dtype=torch.float)

    vocab_size = model_vocab_size if "model_vocab_size" in globals() else 60

    element_indices = []
    for elem in elements:
        idx = element_to_idx.get(elem, 0)
        if idx >= vocab_size:
            idx = 0  # map unseen elements to OTHER
        element_indices.append(idx)
    x = torch.tensor(element_indices, dtype=torch.long)

    adjacency, bond_orders, _ = build_adjacency_and_orders(bonds_data, len(elements))

    bond_types_dict = {}
    for (i, j), order in bond_orders.items():
        if order == 1.0:
            bond_type = 0
        elif order == 1.5:
            bond_type = 1
        elif order == 2.0:
            bond_type = 2
        elif order == 3.0:
            bond_type = 3
        else:
            bond_type = 0
        bond_types_dict[(i, j)] = bond_type

    subgraph_a = dfs_subgraph(adjacency, node_a, depth=subgraph_depth)
    subgraph_b = dfs_subgraph(adjacency, node_b, depth=subgraph_depth)

    subgraph_nodes = subgraph_a.union(subgraph_b)
    subgraph_nodes.add(node_a)
    subgraph_nodes.add(node_b)

    node_list = sorted(list(subgraph_nodes))
    node_mapping = {old: new for new, old in enumerate(node_list)}

    edge_index = []
    edge_attr = []

    for old_i in node_list:
        for old_j in adjacency[old_i]:
            if old_j in node_mapping:
                edge_index.append([node_mapping[old_i], node_mapping[old_j]])
                edge_attr.append(bond_types_dict.get((old_i, old_j), 0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long) if edge_attr else torch.empty((0,), dtype=torch.long)

    data = Data(
        x=x[node_list].to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        pos=pos[node_list].to(device),
        node_a=torch.tensor(node_mapping[node_a], dtype=torch.long).to(device),
        node_b=torch.tensor(node_mapping[node_b], dtype=torch.long).to(device),
    )

    return data


def predict_bond_geometry(model, data):
    with torch.no_grad():
        pred_dist, pred_geom_a, pred_geom_b = model(data)

    return (
        pred_dist.squeeze().cpu().item(),
        pred_geom_a.squeeze().cpu().numpy(),
        pred_geom_b.squeeze().cpu().numpy(),
    )


def to_theta_delta_phi(geom, max_bonds):
    """
    Converts compute_local_geometry output:
        [(r, theta_rad, phi_rad), ...]
    into training target:
        [[theta_deg, delta_phi_deg], ...]
    """
    arr = np.zeros((max_bonds, 2), dtype=float)
    mask = np.zeros(max_bonds, dtype=float)

    prev_phi_deg = None

    for slot_idx, (_, theta, phi) in enumerate(geom[:max_bonds]):
        theta_deg = math.degrees(theta)
        phi_deg = math.degrees(phi)

        if prev_phi_deg is None:
            delta_phi_deg = 0.0
        else:
            delta_phi_deg = (phi_deg - prev_phi_deg) % 360.0

        arr[slot_idx] = [theta_deg, delta_phi_deg]
        mask[slot_idx] = 1.0

        prev_phi_deg = phi_deg

    return arr, mask


def iter_valid_molecules(csv_path, chunksize=5000):
    valid_mol_idx = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        for raw_idx, row in chunk.iterrows():
            try:
                if GEOMETRY_COLUMN not in row or pd.isna(row[GEOMETRY_COLUMN]):
                    continue

                if (
                    "Pubchem_to_xyz_aid_mapping" not in row
                    or pd.isna(row["Pubchem_to_xyz_aid_mapping"])
                    or pubchem_mapping_has_error(row["Pubchem_to_xyz_aid_mapping"])
                ):
                    continue

                xyz_coordinates, _ = parse_xyz_string(row[GEOMETRY_COLUMN])
                if not xyz_coordinates or len(xyz_coordinates) < 2:
                    continue

                coords, elements, bonds = parse_pubchem_data(row, xyz_coordinates)
                if not coords or not elements or not bonds:
                    continue

                cid = row.get("cid", f"molecule_{raw_idx}")

                yield valid_mol_idx, coords, elements, bonds, cid
                valid_mol_idx += 1

            except Exception:
                continue


# =====================================================
# LOCAL TRAINING-STYLE METRICS
# =====================================================
def compute_local_training_style_details(
    model,
    coords,
    elements,
    bonds,
    element_to_idx,
    device,
    max_bonds,
    cid,
    csv_name,
    mol_idx,
):
    coords = np.asarray(coords, dtype=float)
    pos_tensor = torch.tensor(coords, dtype=torch.float, device=device)

    adjacency, bond_orders, bond_id_lookup = build_adjacency_and_orders(bonds, len(elements))

    bond_rows = []
    angle_rows = []

    processed_bonds = set()

    for bond in bonds:
        if "i" not in bond or "j" not in bond:
            continue

        i, j = bond["i"], bond["j"]

        if i >= len(elements) or j >= len(elements) or i == j:
            continue

        bond_key = tuple(sorted((i, j)))
        if bond_key in processed_bonds:
            continue
        processed_bonds.add(bond_key)

        order = bond.get("order", np.nan)
        bond_id = bond_id_lookup.get((i, j), np.nan)

        # =====================================================
        # MODEL PREDICTION FOR THIS BOND
        # =====================================================
        data = create_graph_data(
            coords.tolist(),
            elements,
            bonds,
            element_to_idx,
            i,
            j,
            max_bonds,
            device=device,
        )

        pred_dist, pred_geom_i, pred_geom_j = predict_bond_geometry(model, data)

        true_dist = np.linalg.norm(coords[i] - coords[j])
        dist_error = pred_dist - true_dist

        bond_rows.append({
            "csv": csv_name,
            "mol_idx": mol_idx,
            "cid": cid,
            "bond_id": bond_id,
            "atom_i": i,
            "atom_j": j,
            "element_i": elements[i],
            "element_j": elements[j],
            "bond_type": f"{elements[i]}-{elements[j]}",
            "bond_order": order,
            "true_distance": true_dist,
            "pred_distance": pred_dist,
            "error": dist_error,
            "abs_error": abs(dist_error),
            "squared_error": dist_error ** 2,
            "percent_error": abs(dist_error) / (abs(true_dist) + 1e-3),
        })

        # =====================================================
        # LOCAL GEOMETRY TARGET FOR SIDE i, REFERENCE j
        # =====================================================
        for center, ref, pred_geom in [
            (i, j, pred_geom_i),
            (j, i, pred_geom_j),
        ]:
            other_neighbors = [n for n in adjacency[center] if n != ref]

            if not other_neighbors:
                continue

            sort_keys = get_neighbor_sort_keys(
                other_neighbors,
                elements,
                bond_orders,
                center,
            )

            ordered_neighbors, target_geom_raw, _ = compute_local_geometry(
                pos_tensor,
                center,
                other_neighbors,
                ref,
                neighbor_sort_keys=sort_keys,
            )

            target_geom, mask = to_theta_delta_phi(target_geom_raw, max_bonds)

            for slot_idx, neighbor in enumerate(ordered_neighbors[:max_bonds]):
                if mask[slot_idx] == 0:
                    continue

                true_theta = target_geom[slot_idx, 0]
                true_delta_phi = target_geom[slot_idx, 1]

                pred_theta = float(pred_geom[slot_idx, 0])
                pred_delta_phi = float(pred_geom[slot_idx, 1])

                theta_error = pred_theta - true_theta
                delta_phi_error = pred_delta_phi - true_delta_phi

                ref_order = bond_orders.get((center, ref), np.nan)
                neighbor_order = bond_orders.get((center, neighbor), np.nan)

                angle_rows.append({
                    "csv": csv_name,
                    "mol_idx": mol_idx,
                    "cid": cid,

                    "bond_id": bond_id,
                    "center_atom": center,
                    "reference_atom": ref,
                    "neighbor_atom": neighbor,

                    "center_element": elements[center],
                    "reference_element": elements[ref],
                    "neighbor_element": elements[neighbor],

                    "local_angle_type": (
                        f"{elements[ref]}-{elements[center]}-{elements[neighbor]}"
                    ),

                    "local_angle_order_type": (
                        f"{elements[ref]}-{elements[center]}-{elements[neighbor]} "
                        f"({ref_order}-{neighbor_order})"
                    ),

                    "reference_bond_order": ref_order,
                    "neighbor_bond_order": neighbor_order,

                    "slot_idx": slot_idx,

                    "true_theta": true_theta,
                    "pred_theta": pred_theta,
                    "theta_error": theta_error,
                    "theta_abs_error": abs(theta_error),
                    "theta_squared_error": theta_error ** 2,

                    "true_delta_phi": true_delta_phi,
                    "pred_delta_phi": pred_delta_phi,
                    "delta_phi_error": delta_phi_error,
                    "delta_phi_abs_error": abs(delta_phi_error),
                    "delta_phi_squared_error": delta_phi_error ** 2,

                    "training_style_theta_loss": abs(theta_error) / 180.0,
                    "training_style_delta_phi_loss": abs(delta_phi_error) / 360.0,
                    "training_style_geometry_loss": (
                        (abs(theta_error) / 180.0) +
                        (abs(delta_phi_error) / 360.0)
                    ) / 2.0,
                })

    return bond_rows, angle_rows


# =====================================================
# MAIN
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Local training-style bond + angle metrics")
    parser.add_argument("--root", type=Path, required=True,
                        help="Root directory of the project (contains checkpoints/, config/)")
    parser.add_argument("--csv-dir", type=Path, required=True,
                        help="Directory containing input CSV files")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint (default: <root>/checkpoints/best_model.pt)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config YAML (default: <root>/config/config.yaml)")
    parser.add_argument("--out-bond-csv", type=Path, default=None,
                        help="Output bond details CSV (default: <root>/local_training_style_bond_details.csv)")
    parser.add_argument("--out-angle-csv", type=Path, default=None,
                        help="Output angle details CSV (default: <root>/local_training_style_angle_details.csv)")
    parser.add_argument("--out-summary", type=Path, default=None,
                        help="Output summary TXT (default: <root>/local_training_style_summary.txt)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root

    checkpoint_path = args.checkpoint or root / "checkpoints" / "best_model.pt"
    config_path     = args.config      or root / "config" / "config.yaml"
    csv_dir         = args.csv_dir
    out_bond_csv    = args.out_bond_csv   or root / "local_training_style_bond_details.csv"
    out_angle_csv   = args.out_angle_csv  or root / "local_training_style_angle_details.csv"
    out_summary_txt = args.out_summary    or root / "local_training_style_summary.txt"

    print("=" * 70)
    print("LOCAL TRAINING-STYLE BOND + ANGLE METRICS")
    print("No reconstruction. No alignment.")
    print("=" * 70)

    model, element_to_idx, device, config = load_trained_model(
        checkpoint_path,
        config_path,
    )

    global model_vocab_size
    model_vocab_size = model.element_embedding.weight.shape[0]

    max_bonds = config["data"]["max_bonds"]

    for p in [out_bond_csv, out_angle_csv, out_summary_txt]:
        if p.exists():
            p.unlink()

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")

    bond_buffer = []
    angle_buffer = []

    total_bonds = 0
    total_angles = 0

    for csv_path in tqdm(csv_files, desc="CSV files"):
        for mol_idx, coords, elements, bonds, cid in tqdm(
            iter_valid_molecules(csv_path),
            desc=csv_path.name,
            leave=False,
        ):
            bond_rows, angle_rows = compute_local_training_style_details(
                model=model,
                coords=coords,
                elements=elements,
                bonds=bonds,
                element_to_idx=element_to_idx,
                device=device,
                max_bonds=max_bonds,
                cid=cid,
                csv_name=csv_path.name,
                mol_idx=mol_idx,
            )

            bond_buffer.extend(bond_rows)
            angle_buffer.extend(angle_rows)

            total_bonds += len(bond_rows)
            total_angles += len(angle_rows)

            if len(bond_buffer) >= FLUSH_EVERY:
                pd.DataFrame(bond_buffer).to_csv(
                    out_bond_csv,
                    mode="a",
                    header=not out_bond_csv.exists(),
                    index=False,
                )
                bond_buffer.clear()

            if len(angle_buffer) >= FLUSH_EVERY:
                pd.DataFrame(angle_buffer).to_csv(
                    out_angle_csv,
                    mode="a",
                    header=not out_angle_csv.exists(),
                    index=False,
                )
                angle_buffer.clear()

    if bond_buffer:
        pd.DataFrame(bond_buffer).to_csv(
            out_bond_csv,
            mode="a",
            header=not out_bond_csv.exists(),
            index=False,
        )

    if angle_buffer:
        pd.DataFrame(angle_buffer).to_csv(
            out_angle_csv,
            mode="a",
            header=not out_angle_csv.exists(),
            index=False,
        )

    # =====================================================
    # SUMMARY
    # =====================================================
    lines = []
    lines.append("=" * 70)
    lines.append("LOCAL TRAINING-STYLE SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total bonds evaluated:  {total_bonds}")
    lines.append(f"Total local angles evaluated: {total_angles}")
    lines.append(f"Bond CSV:  {out_bond_csv}")
    lines.append(f"Angle CSV: {out_angle_csv}")

    if out_bond_csv.exists():
        bdf = pd.read_csv(out_bond_csv)

        lines.append("")
        lines.append("BOND SUMMARY BY TYPE + ORDER")
        lines.append("-" * 70)

        bond_summary = (
            bdf.groupby(["bond_type", "bond_order"])
            .agg(
                count=("error", "size"),
                mae=("abs_error", "mean"),
                rmse=("squared_error", lambda x: np.sqrt(x.mean())),
                mean_error=("error", "mean"),
                mean_percent_error=("percent_error", "mean"),
            )
            .sort_values("mae", ascending=False)
        )

        lines.append(bond_summary.round(6).to_string())

    if out_angle_csv.exists():
        adf = pd.read_csv(out_angle_csv)

        lines.append("")
        lines.append("LOCAL ANGLE SUMMARY BY TYPE + ORDER")
        lines.append("-" * 70)

        angle_summary = (
            adf.groupby("local_angle_order_type")
            .agg(
                count=("theta_error", "size"),
                theta_mae=("theta_abs_error", "mean"),
                theta_rmse=("theta_squared_error", lambda x: np.sqrt(x.mean())),
                delta_phi_mae=("delta_phi_abs_error", "mean"),
                delta_phi_rmse=("delta_phi_squared_error", lambda x: np.sqrt(x.mean())),
                geometry_loss=("training_style_geometry_loss", "mean"),
            )
            .sort_values("geometry_loss", ascending=False)
        )

        lines.append(angle_summary.round(6).to_string())

    summary_text = "\n".join(lines)

    with open(out_summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSaved summary to: {out_summary_txt}")


if __name__ == "__main__":
    main()