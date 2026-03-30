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
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from dataset import (
    parse_xyz_string,
    parse_pubchem_data,
    _pubchem_mapping_has_error,
    parse_bonds_from_dataset,
)

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


def main():
    parser = argparse.ArgumentParser(description='Visualize nth molecule from dataset CSV as HTML')
    parser.add_argument('--csv', default='Compound_000000001_000025000_props_geom.csv',
                        help='Path to CSV file (default: Compound_000000001_000025000_props_geom.csv)')
    parser.add_argument('--row', type=int, default=0,
                        help='0-based row index to visualise (default: 0)')
    parser.add_argument('--out', default=None,
                        help='Output HTML path (default: molecule_row<N>.html)')
    args = parser.parse_args()

    out = args.out or f'molecule_row{args.row}.html'

    print(f'Loading row {args.row} from {args.csv} ...')
    coords, elements, bonds, cid, source = load_row(args.csv, args.row)

    print(f'  CID={cid}  atoms={len(elements)}  bonds={len(bonds)}  source={source}')

    fig = build_figure(coords, elements, bonds, cid, source)
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'Saved -> {out}')

    # Also open in browser
    import webbrowser, pathlib
    webbrowser.open(pathlib.Path(out).resolve().as_uri())


if __name__ == '__main__':
    main()
