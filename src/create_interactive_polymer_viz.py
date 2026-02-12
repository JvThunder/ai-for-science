#!/usr/bin/env python3
"""
Create interactive 3D visualization of polymer reconstructions using Plotly.

Generates an HTML file with interactive 3D molecular viewer that supports:
- Rotation (click and drag)
- Zoom (scroll wheel)
- Pan (shift + drag)
- Multiple views side-by-side
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path


def read_xyz(filepath):
    """Read XYZ file and return elements and positions."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    elements = []
    positions = []

    for line in lines[2:2+n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return elements, np.array(positions), comment


def get_bonds_from_topology(elements, n_carbons):
    """
    Reconstruct bond topology from polyethylene structure.

    For polyethylene: Each carbon has 2 hydrogens and bonds to next carbon.
    """
    bonds = []
    atom_idx = 0

    for c_idx in range(n_carbons):
        c_atom_idx = atom_idx
        h1_idx = atom_idx + 1
        h2_idx = atom_idx + 2

        # C-H bonds
        bonds.append((c_atom_idx, h1_idx, 'C-H'))
        bonds.append((c_atom_idx, h2_idx, 'C-H'))

        # C-C bond (except last)
        if c_idx < n_carbons - 1:
            next_c_idx = c_atom_idx + 3
            bonds.append((c_atom_idx, next_c_idx, 'C-C'))

        atom_idx += 3

    return bonds


def create_interactive_molecule(elements, positions, bonds, title,
                                show_labels=True, subplot_title=None):
    """
    Create interactive 3D molecular plot using Plotly.

    Returns list of traces for adding to figure.
    """
    traces = []

    # CPK colors
    element_colors = {
        'H': '#FFFFFF',
        'C': '#909090',
        'N': '#3050F8',
        'O': '#FF0D0D'
    }

    element_sizes = {
        'H': 8,
        'C': 15,
        'N': 15,
        'O': 15
    }

    # Draw bonds first (so atoms appear on top)
    bond_x, bond_y, bond_z = [], [], []
    bond_colors = []

    for i, j, bond_type in bonds:
        # Add bond line
        bond_x.extend([positions[i, 0], positions[j, 0], None])
        bond_y.extend([positions[i, 1], positions[j, 1], None])
        bond_z.extend([positions[i, 2], positions[j, 2], None])

    # C-C bonds (thick, dark)
    cc_x, cc_y, cc_z = [], [], []
    ch_x, ch_y, ch_z = [], [], []

    for i, j, bond_type in bonds:
        if bond_type == 'C-C':
            cc_x.extend([positions[i, 0], positions[j, 0], None])
            cc_y.extend([positions[i, 1], positions[j, 1], None])
            cc_z.extend([positions[i, 2], positions[j, 2], None])
        else:  # C-H
            ch_x.extend([positions[i, 0], positions[j, 0], None])
            ch_y.extend([positions[i, 1], positions[j, 1], None])
            ch_z.extend([positions[i, 2], positions[j, 2], None])

    # Add C-C bonds (thick)
    if cc_x:
        traces.append(go.Scatter3d(
            x=cc_x, y=cc_y, z=cc_z,
            mode='lines',
            line=dict(color='#404040', width=8),
            hoverinfo='skip',
            showlegend=False,
            name='C-C bonds'
        ))

    # Add C-H bonds (thin)
    if ch_x:
        traces.append(go.Scatter3d(
            x=ch_x, y=ch_y, z=ch_z,
            mode='lines',
            line=dict(color='#808080', width=4),
            hoverinfo='skip',
            showlegend=False,
            name='C-H bonds'
        ))

    # Draw atoms
    for elem in set(elements):
        indices = [i for i, e in enumerate(elements) if e == elem]
        elem_positions = positions[indices]

        hover_text = [f"{elem}{i}<br>({positions[i,0]:.2f}, {positions[i,1]:.2f}, {positions[i,2]:.2f})"
                     for i in indices]

        traces.append(go.Scatter3d(
            x=elem_positions[:, 0],
            y=elem_positions[:, 1],
            z=elem_positions[:, 2],
            mode='markers',
            marker=dict(
                size=element_sizes.get(elem, 10),
                color=element_colors.get(elem, '#FF1493'),
                line=dict(color='black', width=0.5),
                opacity=0.95
            ),
            text=hover_text,
            hoverinfo='text',
            name=f'{elem} atoms',
            legendgroup=elem
        ))

    return traces


def create_comparison_plot(xyz_simple, xyz_chunked, n_carbons, output_file):
    """
    Create interactive comparison plot with side-by-side views.
    """
    # Read both files
    elements_simple, positions_simple, comment_simple = read_xyz(xyz_simple)
    elements_chunked, positions_chunked, comment_chunked = read_xyz(xyz_chunked)

    # Get bonds
    bonds = get_bonds_from_topology(elements_simple, n_carbons)

    # Create subplots (1 row, 2 columns)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Simple BFS Reconstruction', 'Chunk-Based Reconstruction'),
        horizontal_spacing=0.05
    )

    # Add simple reconstruction
    traces_simple = create_interactive_molecule(
        elements_simple, positions_simple, bonds,
        title='Simple BFS',
        subplot_title='Simple'
    )

    for trace in traces_simple:
        fig.add_trace(trace, row=1, col=1)

    # Add chunked reconstruction
    traces_chunked = create_interactive_molecule(
        elements_chunked, positions_chunked, bonds,
        title='Chunked',
        subplot_title='Chunked'
    )

    for trace in traces_chunked:
        # Don't show legend for second plot (duplicate)
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive Polyethylene Reconstruction Comparison ({n_carbons} Carbons)<br>' +
                 '<sub>Click and drag to rotate | Scroll to zoom | Shift+drag to pan</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.05,
            xanchor='center',
            yanchor='top',
            orientation='h'
        ),
        height=700,
        hovermode='closest',
        template='plotly_white'
    )

    # Update 3D scene settings for both subplots
    scene_settings = dict(
        xaxis=dict(
            title='X (Å)',
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white',
            showbackground=True,
        ),
        yaxis=dict(
            title='Y (Å)',
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white',
            showbackground=True,
        ),
        zaxis=dict(
            title='Z (Å)',
            backgroundcolor='rgb(240, 240, 240)',
            gridcolor='white',
            showbackground=True,
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0)
        ),
        aspectmode='data'
    )

    fig.update_scenes(scene_settings)

    # Save to HTML
    fig.write_html(
        output_file,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'polymer_{n_carbons}c_comparison',
                'height': 1000,
                'width': 1800,
                'scale': 2
            }
        }
    )

    print(f"\n✓ Interactive visualization saved to: {output_file}")
    print(f"  Open in browser to explore:")
    print(f"    - Rotate: Click and drag")
    print(f"    - Zoom: Scroll wheel")
    print(f"    - Pan: Shift + drag")
    print(f"    - Reset view: Double-click")
    print(f"    - Hover over atoms for details")


def create_single_plot(xyz_file, n_carbons, output_file, title="Polymer Reconstruction"):
    """Create interactive plot for a single structure."""
    elements, positions, comment = read_xyz(xyz_file)
    bonds = get_bonds_from_topology(elements, n_carbons)

    fig = go.Figure()

    traces = create_interactive_molecule(
        elements, positions, bonds,
        title=title
    )

    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>{comment}</sub><br>' +
                 '<sub>Click and drag to rotate | Scroll to zoom | Shift+drag to pan</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='bottom',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=800,
        scene=dict(
            xaxis=dict(
                title='X (Å)',
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
            ),
            yaxis=dict(
                title='Y (Å)',
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
            ),
            zaxis=dict(
                title='Z (Å)',
                backgroundcolor='rgb(240, 240, 240)',
                gridcolor='white',
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='data'
        ),
        hovermode='closest',
        template='plotly_white'
    )

    fig.write_html(
        output_file,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'polymer_{n_carbons}c',
                'height': 1200,
                'width': 1200,
                'scale': 2
            }
        }
    )

    print(f"\n✓ Interactive visualization saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive 3D visualization of polymer reconstructions'
    )
    parser.add_argument('--simple', type=str, required=True,
                       help='Path to simple reconstruction XYZ file')
    parser.add_argument('--chunked', type=str, default=None,
                       help='Path to chunked reconstruction XYZ file (optional, for comparison)')
    parser.add_argument('--n_carbons', type=int, required=True,
                       help='Number of carbons in chain')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file (default: auto-generated)')
    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        if args.chunked:
            args.output = f'polymer_results/interactive_comparison_{args.n_carbons}c.html'
        else:
            args.output = f'polymer_results/interactive_{args.n_carbons}c.html'

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Creating Interactive 3D Visualization")
    print("="*60)
    print(f"Simple XYZ: {args.simple}")
    if args.chunked:
        print(f"Chunked XYZ: {args.chunked}")
    print(f"Output: {args.output}")

    # Create visualization
    if args.chunked:
        create_comparison_plot(args.simple, args.chunked, args.n_carbons, args.output)
    else:
        create_single_plot(args.simple, args.n_carbons, args.output)

    print("\n" + "="*60)
    print("Done! Open the HTML file in your browser.")
    print("="*60)


if __name__ == '__main__':
    main()
