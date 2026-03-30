#!/usr/bin/env python3
"""
Visualize the 3D geometry of the nth row from the dataset CSV as an interactive HTML.

Usage:
    python visualize_molecule.py                    # row 0 from default CSV
    python visualize_molecule.py --row 5            # row 5
    python visualize_molecule.py --row 2 --csv path/to/file.csv
    python visualize_molecule.py --row 0 --out molecule.html
"""

import argparse
import sys
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from dataset import (
    parse_xyz_string,
    parse_pubchem_data,
    _pubchem_mapping_has_error,
    parse_bonds_from_dataset,
)
from geometry import compute_local_geometry, convert_to_relative_azimuthal, get_neighbor_sort_keys

import plotly.graph_objects as go

# CPK-style element colours
ELEMENT_COLORS = {
    'H':  '#FFFFFF', 'C':  '#404040', 'N':  '#3050F8', 'O':  '#FF0D0D',
    'F':  '#90E050', 'S':  '#FFFF30', 'Cl': '#1FF01F', 'Br': '#A62929',
    'P':  '#FF8000', 'I':  '#940094', 'B':  '#FFB5B5', 'Si': '#F0C8A0',
    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Ca': '#3DFF00',
    'Fe': '#E06633', 'Cu': '#C88033', 'Zn': '#7D80B0',
}
DEFAULT_COLOR = '#FF1493'

# Van-der-Waals radii (Å) scaled for marker size
ELEMENT_RADII = {
    'H': 0.53, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'S': 1.02, 'Cl': 0.99, 'Br': 1.14, 'P': 1.06, 'I': 1.33,
    'B': 0.87, 'Si': 1.17,
}
DEFAULT_RADIUS = 0.85

# Bond-order → dash style
BOND_DASH = {1.0: 'solid', 1.5: 'dot', 2.0: 'dash', 3.0: 'dashdot'}
BOND_WIDTH = {1.0: 3, 1.5: 2.5, 2.0: 4, 3.0: 5}


def load_row(csv_path: str, row_idx: int):
    """Load and parse the nth row using PubChem priority, XYZ fallback."""
    df = pd.read_csv(csv_path, skiprows=range(1, row_idx + 1), nrows=1)
    if df.empty:
        raise IndexError(f"Row {row_idx} not found in {csv_path}")
    row = df.iloc[0]

    if pd.isna(row.get('geometry_xyz', float('nan'))):
        raise ValueError(f"Row {row_idx} has no geometry_xyz")

    xyz_coords, xyz_elems = parse_xyz_string(row['geometry_xyz'])

    use_pubchem = (
        'Pubchem_to_xyz_aid_mapping' in row
        and not _pubchem_mapping_has_error(row.get('Pubchem_to_xyz_aid_mapping'))
    )

    if use_pubchem:
        coords, elements, bonds = parse_pubchem_data(row, xyz_coords)
        source = 'PubChem'
        if not coords or not bonds:
            coords, elements = xyz_coords, xyz_elems
            bonds = parse_bonds_from_dataset(row['bonds_geom_valence'])
            source = 'XYZ (PubChem fallback)'
    else:
        coords, elements = xyz_coords, xyz_elems
        bonds = parse_bonds_from_dataset(row['bonds_geom_valence'])
        source = 'XYZ'

    cid = row.get('cid', f'row_{row_idx}')
    return np.array(coords), elements, bonds, cid, source


def build_figure(coords: np.ndarray, elements, bonds, cid, source: str) -> go.Figure:
    fig = go.Figure()

    # --- Bond traces (one per bond so we get per-bond styling) ---
    for bond in bonds:
        i, j = bond['i'], bond['j']
        if i >= len(coords) or j >= len(coords):
            continue
        order = bond.get('order', 1.0)
        mid = (coords[i] + coords[j]) / 2
        color = '#888888'
        dash  = BOND_DASH.get(order, 'solid')
        width = BOND_WIDTH.get(order, 3)
        label = {1.0: 'single', 1.5: 'aromatic', 2.0: 'double', 3.0: 'triple'}.get(order, f'order {order}')

        fig.add_trace(go.Scatter3d(
            x=[coords[i, 0], mid[0], coords[j, 0]],
            y=[coords[i, 1], mid[1], coords[j, 1]],
            z=[coords[i, 2], mid[2], coords[j, 2]],
            mode='lines',
            line=dict(color=color, width=width, dash=dash),
            hoverinfo='text',
            hovertext=f'{elements[i]}{i}–{elements[j]}{j} ({label})<br>'
                      f'd = {np.linalg.norm(coords[j]-coords[i]):.3f} Å',
            name=label,
            legendgroup=label,
            showlegend=False,
        ))

    # --- Atom traces (grouped by element for legend) ---
    by_element: dict = {}
    for idx, (elem, pos) in enumerate(zip(elements, coords)):
        by_element.setdefault(elem, {'x': [], 'y': [], 'z': [], 'text': []})
        by_element[elem]['x'].append(pos[0])
        by_element[elem]['y'].append(pos[1])
        by_element[elem]['z'].append(pos[2])
        by_element[elem]['text'].append(
            f'<b>{elem}{idx}</b><br>x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}'
        )

    for elem, data in sorted(by_element.items()):
        color  = ELEMENT_COLORS.get(elem, DEFAULT_COLOR)
        radius = ELEMENT_RADII.get(elem, DEFAULT_RADIUS)
        size   = max(8, int(radius * 22))
        fig.add_trace(go.Scatter3d(
            x=data['x'], y=data['y'], z=data['z'],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(color='#333333', width=1),
                opacity=0.95,
            ),
            text=[f'<b>{elem}</b>'] * len(data['x']),
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=data['text'],
            hoverinfo='text',
            name=elem,
            legendgroup=elem,
        ))

    # Bond-order legend entries
    for order, label in [(1.0,'single'), (1.5,'aromatic'), (2.0,'double'), (3.0,'triple')]:
        if any(b.get('order', 1.0) == order for b in bonds):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='lines',
                line=dict(color='#888888', width=BOND_WIDTH[order], dash=BOND_DASH[order]),
                name=f'{label} bond',
                legendgroup=label,
                showlegend=True,
            ))

    fig.update_layout(
        title=dict(
            text=f'CID {cid} — {len(elements)} atoms, {len(bonds)} bonds '
                 f'<span style="font-size:12px;color:gray">({source})</span>',
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(title='X (Å)', showgrid=True, zeroline=False),
            yaxis=dict(title='Y (Å)', showgrid=True, zeroline=False),
            zaxis=dict(title='Z (Å)', showgrid=True, zeroline=False),
            bgcolor='#f7f7f7',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(
            itemsizing='constant',
            title='Elements / Bonds',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='white',
    )
    return fig


# ── bond-order encoding matching dataset.py ───────────────────────────────────
_ORDER_TO_TYPE  = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}
_TYPE_TO_ORDER  = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
_ORDER_LABEL    = {1.0: 'single', 1.5: 'aromatic', 2.0: 'double', 3.0: 'triple'}
MAX_BONDS       = 7


def _compute_bond_geometries(
    coords: np.ndarray,
    elements: List[str],
    bonds: List[Dict],
) -> List[Dict]:
    """
    For every bond (i, j) compute ground-truth geometry targets identical to
    what _process_molecule_bond stores as target_distance / target_geometry_a/b.

    Returns a list of dicts, one per bond:
        bond_idx, i, j, elem_i, elem_j, order,
        distance,
        geom_a  [(theta_deg, delta_phi_deg), ...],  (node i side)
        geom_b  [(theta_deg, delta_phi_deg), ...],  (node j side)
        n_neighbors_a, n_neighbors_b
    """
    import torch
    pos = torch.tensor(coords, dtype=torch.float)

    # Build adjacency and bond-type dicts
    adjacency    = defaultdict(list)
    bond_types   = {}   # (i,j) -> bond_type int

    valid_bonds = []
    for bond in bonds:
        i, j = bond['i'], bond['j']
        if i >= len(coords) or j >= len(coords) or i == j:
            continue
        order = bond.get('order', 1.0)
        btype = _ORDER_TO_TYPE.get(order, 0)
        adjacency[i].append(j)
        adjacency[j].append(i)
        bond_types[(i, j)] = btype
        bond_types[(j, i)] = btype
        valid_bonds.append((i, j, order))

    bond_orders_dict = {(i, j): _TYPE_TO_ORDER[t] for (i, j), t in bond_types.items()}

    results = []
    for bond_idx, (ni, nj, order) in enumerate(valid_bonds):
        dist = float(torch.norm(pos[ni] - pos[nj]).item())

        def _geom_for(center, ref, adj):
            neighbors = list(adj[center])
            if ref not in neighbors:
                return []
            sort_keys = get_neighbor_sort_keys(neighbors, elements, bond_orders_dict, center)
            ordered, geometry, _ = compute_local_geometry(
                pos, center_idx=center, neighbor_indices=neighbors,
                reference_idx=ref, augment_rotation=False,
                neighbor_sort_keys=sort_keys,
            )
            # geometry is list of (r, theta_rad, phi_rad)
            # convert to degrees and then to relative delta-phi
            geo_deg = []
            for k in range(MAX_BONDS):
                if k < len(geometry):
                    r, theta, phi = geometry[k]
                    geo_deg.append([r, math.degrees(theta), math.degrees(phi)])
                else:
                    geo_deg.append([0.0, 0.0, 0.0])
            geo_np  = np.array(geo_deg)
            mask    = np.array([1.0 if k < len(geometry) else 0.0 for k in range(MAX_BONDS)])
            geo_rel = convert_to_relative_azimuthal(geo_np, mask)  # (MAX_BONDS, 3)
            # return only valid (theta, delta_phi) pairs
            out = []
            for k in range(MAX_BONDS):
                if mask[k]:
                    out.append((geo_rel[k, 1], geo_rel[k, 2]))  # (theta_deg, delta_phi_deg)
            return out

        geom_a = _geom_for(ni, nj, adjacency)
        geom_b = _geom_for(nj, ni, adjacency)

        results.append(dict(
            bond_idx=bond_idx,
            i=ni, j=nj,
            elem_i=elements[ni], elem_j=elements[nj],
            order=order,
            distance=dist,
            geom_a=geom_a,
            geom_b=geom_b,
            n_neighbors_a=len(adjacency[ni]),
            n_neighbors_b=len(adjacency[nj]),
        ))

    return results


def print_losses(
    bond_geoms: List[Dict],
    distance_weight: float = 1.5,
    geometry_weight: float = 1.0,
):
    """
    Print per-bond ground-truth geometry targets and the loss a naive mean
    predictor would incur.  The 'naive' baseline predicts:
        distance  = mean bond distance across all bonds in this molecule
        theta     = 109.5° (tetrahedral)
        delta_phi = 120°   (equal spacing)

    Losses match the formulas in utils/loss.py:
        distance_loss = |pred - target| / (|target| + 1e-3)   (percentage loss)
        theta_loss    = |pred - target| / 180
        phi_loss      = |pred - target| / 360
        geometry_loss = (theta_loss + phi_loss) / 2
    """
    if not bond_geoms:
        print('  No valid bonds found.')
        return

    distances = [b['distance'] for b in bond_geoms]
    mean_dist = float(np.mean(distances))

    NAIVE_THETA    = 109.5
    NAIVE_PHI      = 120.0
    NAIVE_DIST     = mean_dist

    header = (f"{'#':>3}  {'Bond':>10}  {'Ord':>4}  {'Dist(A)':>8}  "
              f"{'dist_loss':>9}  {'geom_a_loss':>11}  {'geom_b_loss':>11}  "
              f"{'nA':>3}  {'nB':>3}")
    print()
    print('  Ground-truth geometry targets (naive baseline = tetrahedral predictor)')
    print('  ' + '-' * (len(header) - 2))
    print('  ' + header)
    print('  ' + '-' * (len(header) - 2))

    total_dist_loss = 0.0
    total_geom_loss = 0.0

    for b in bond_geoms:
        # Distance loss
        dloss = abs(NAIVE_DIST - b['distance']) / (abs(b['distance']) + 1e-3)

        # Geometry loss for side A
        def _side_loss(geom):
            if not geom:
                return float('nan')
            tl = sum(abs(t - NAIVE_THETA) / 180.0 for t, _ in geom) / len(geom)
            pl = sum(abs(dp - NAIVE_PHI)  / 360.0 for _, dp in geom) / len(geom)
            return (tl + pl) / 2.0

        gl_a = _side_loss(b['geom_a'])
        gl_b = _side_loss(b['geom_b'])
        gl   = np.nanmean([gl_a, gl_b])

        total_dist_loss += dloss
        total_geom_loss += gl if not math.isnan(gl) else 0.0

        label = f"{b['elem_i']}{b['i']}-{b['elem_j']}{b['j']}"
        ord_s = _ORDER_LABEL.get(b['order'], str(b['order']))[:4]
        gl_a_s = f'{gl_a:.4f}' if not math.isnan(gl_a) else '   n/a'
        gl_b_s = f'{gl_b:.4f}' if not math.isnan(gl_b) else '   n/a'
        print(f"  {b['bond_idx']:>3}  {label:>10}  {ord_s:>4}  "
              f"{b['distance']:>8.4f}  {dloss:>9.4f}  {gl_a_s:>11}  {gl_b_s:>11}  "
              f"{b['n_neighbors_a']:>3}  {b['n_neighbors_b']:>3}")

    n = len(bond_geoms)
    print('  ' + '-' * (len(header) - 2))

    avg_dist_loss = total_dist_loss / n
    avg_geom_loss = total_geom_loss / n
    weighted_total = distance_weight * avg_dist_loss + geometry_weight * avg_geom_loss

    print(f'\n  Summary ({n} bonds) — naive baseline vs ground truth:')
    print(f'    mean distance       : {mean_dist:.4f} A')
    print(f'    distance_loss (avg) : {avg_dist_loss:.4f}   (weight {distance_weight})')
    print(f'    geometry_loss (avg) : {avg_geom_loss:.4f}   (weight {geometry_weight})')
    print(f'    total_loss          : {weighted_total:.4f}   '
          f'= {distance_weight}*{avg_dist_loss:.4f} + {geometry_weight}*{avg_geom_loss:.4f}')
    print()


def main():
    parser = argparse.ArgumentParser(description='Visualize nth molecule from dataset CSV as HTML')
    parser.add_argument('--csv', default='Compound_000000001_000025000_props_geom.csv',
                        help='Path to CSV file (default: Compound_000000001_000025000_props_geom.csv)')
    parser.add_argument('--row', type=int, default=0,
                        help='0-based row index to visualise (default: 0)')
    parser.add_argument('--out', default=None,
                        help='Output HTML path (default: molecule_row<N>.html)')
    parser.add_argument('--distance-weight', type=float, default=1.5,
                        help='Loss weight for distance (matches config loss.distance_weight, default: 1.5)')
    parser.add_argument('--geometry-weight', type=float, default=1.0,
                        help='Loss weight for geometry (matches config loss.geometry_weight, default: 1.0)')
    args = parser.parse_args()

    out = args.out or f'molecule_row{args.row}.html'

    print(f'Loading row {args.row} from {args.csv} ...')
    coords, elements, bonds, cid, source = load_row(args.csv, args.row)

    print(f'  CID={cid}  atoms={len(elements)}  bonds={len(bonds)}  source={source}')

    bond_geoms = _compute_bond_geometries(coords, elements, bonds)
    print_losses(bond_geoms, args.distance_weight, args.geometry_weight)

    fig = build_figure(coords, elements, bonds, cid, source)
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'Saved -> {out}')

    # Also open in browser
    import webbrowser, pathlib
    webbrowser.open(pathlib.Path(out).resolve().as_uri())


if __name__ == '__main__':
    main()
