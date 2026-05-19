"""
Multi-file streaming dataset for TB-scale molecular geometry data.

This module provides a streaming dataset implementation that:
- Handles multiple CSV files (TB-scale data)
- Streams data from disk with NO RAM caching
- Uses file-based epochs (each file = 1 epoch)
- Reads CSV in chunks for memory efficiency
- Integrates with stereochemistry-aware geometry computation
- Extracts bond types (single, aromatic, double, triple) as edge features

Key Classes:
- MultiFileStreamingDataset: Main streaming dataset for training

Subgraph Extraction:
    Uses dual extraction strategy (from both bond endpoints) + union to capture
    local context for both cyclic and linear molecular structures. At depth=1:
    - Cyclic structures (triangles, rings): Captures complete local cycles
    - Linear chains: Captures 2-hop neighborhoods (context from both sides)
    - Bond types: Preserved as edge_attr for all edges in subgraph
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from collections import defaultdict, deque, OrderedDict
from typing import List, Dict, Optional, Tuple
import json
import glob
import random
import math
from tqdm import tqdm
import sys
import os
import psutil

# Import our geometry utilities
sys.path.append(os.path.dirname(__file__))
from geometry import compute_local_geometry, convert_to_relative_azimuthal, get_neighbor_sort_keys, compute_chirality_sign

# Periodic table: atomic number → element symbol (covers common chemistry elements)
ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn',
}

GEOMETRY_COLUMN = "DFT_PBE_USPP_Optimized_Geometry"

def _pubchem_mapping_has_error(mapping_val) -> bool:
    """Return True if Pubchem_to_xyz_aid_mapping contains the word 'error' (case-insensitive)."""
    if mapping_val is None:
        return True
    try:
        if pd.isna(mapping_val):
            return True
    except (TypeError, ValueError):
        pass
    return 'error' in str(mapping_val).lower()


def _parse_json_list(value):
    """Parse a JSON-encoded list column, returning the list or None on failure."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_pubchem_data(row, xyz_coordinates: List[List[float]]) -> Tuple[List[List[float]], List[str], List[Dict]]:
    """
    Parse PubChem columns to produce coordinates (from XYZ mapped via aid mapping),
    element symbols, and bond dicts compatible with the existing processing pipeline.

    Args:
        row: DataFrame row with PubChem columns
        xyz_coordinates: 3D coordinates already parsed from GEOMETRY_COLUMN

    Returns:
        coordinates: List of [x, y, z] in PubChem atom ordering (mapped from XYZ)
        elements: List of element symbols in PubChem atom ordering
        bonds: List of {'i': int, 'j': int, 'order': float} (0-indexed positions)
    """
    aids = _parse_json_list(row['atoms.aid_pubchem'])
    atomic_nums = _parse_json_list(row['atoms.element_pubchem'])
    mapping = _parse_json_list(row['Pubchem_to_xyz_aid_mapping'])

    if aids is None or atomic_nums is None or mapping is None:
        return [], [], []
    if len(aids) != len(atomic_nums) or len(mapping) != len(aids):
        return [], [], []

    # Convert atomic numbers to element symbols
    elements = [ATOMIC_NUM_TO_SYMBOL.get(int(z), 'OTHER') for z in atomic_nums]

    # Map 3D coordinates from XYZ ordering to PubChem ordering
    # mapping[i] = XYZ atom ID (1-indexed) for PubChem atom at position i
    n = len(aids)
    coordinates = []
    for i in range(n):
        xyz_idx = int(mapping[i]) - 1  # Convert to 0-indexed
        if 0 <= xyz_idx < len(xyz_coordinates):
            coordinates.append(xyz_coordinates[xyz_idx])
        else:
            return [], [], []  # Mapping out of range — abort

    # Build AID → position mapping for bond index lookup
    aid_to_pos = {int(aid): pos for pos, aid in enumerate(aids)}

    # Parse bond lists
    aid1_list = _parse_json_list(row['bonds.aid1_pubchem'])
    aid2_list = _parse_json_list(row['bonds.aid2_pubchem'])
    order_list = _parse_json_list(row['bonds.order_pubchem'])

    if aid1_list is None or aid2_list is None or order_list is None:
        return coordinates, elements, []

    bonds = []
    for aid1, aid2, order in zip(aid1_list, aid2_list, order_list):
        i = aid_to_pos.get(int(aid1))
        j = aid_to_pos.get(int(aid2))
        if i is None or j is None or i == j:
            continue
        # PubChem bond orders: 1=single, 2=double, 3=triple, 4=aromatic
        bond_order = float(order)
        if bond_order == 4.0:
            bond_order = 1.5  # Aromatic
        bonds.append({'i': i, 'j': j, 'order': bond_order})

    return coordinates, elements, bonds


def parse_xyz_string(xyz_string: str) -> Tuple[List[List[float]], List[str]]:
    """
    Parse XYZ format string to extract atom coordinates and elements.

    Args:
        xyz_string: XYZ format string (possibly JSON-encoded)

    Returns:
        coordinates: List of [x, y, z] coordinates
        elements: List of element symbols
    """
    # Handle JSON encoding
    if isinstance(xyz_string, str):
        if xyz_string.startswith('"') and xyz_string.endswith('"'):
            try:
                xyz_string = json.loads(xyz_string)
            except json.JSONDecodeError:
                xyz_string = xyz_string[1:-1]
                xyz_string = xyz_string.replace('\\"', '"').replace('\\\\', '\\')
        xyz_string = xyz_string.replace('\\n', '\n')

    lines = xyz_string.strip().split('\n')
    if len(lines) < 3:
        return [], []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        return [], []

    coordinates = []
    elements = []

    for i in range(2, min(2 + num_atoms, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            element = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                coordinates.append([x, y, z])
                elements.append(element)
            except ValueError:
                continue

    return coordinates, elements


def parse_bonds_from_dataset(bonds_string) -> List[Dict]:
    """
    Parse bonds_geom_valence JSON string to extract bond information.

    Args:
        bonds_string: JSON string or list containing bond data

    Returns:
        List of bond dictionaries with 'i' and 'j' keys
    """
    if isinstance(bonds_string, str):
        if bonds_string.startswith('"') and bonds_string.endswith('"'):
            try:
                bonds_string = json.loads(bonds_string)
            except json.JSONDecodeError:
                bonds_string = bonds_string[1:-1]

        # Handle escaped JSON
        if isinstance(bonds_string, str):
            try:
                bonds_data = json.loads(bonds_string)
            except json.JSONDecodeError:
                return []
        else:
            bonds_data = bonds_string
    else:
        bonds_data = bonds_string

    return bonds_data if isinstance(bonds_data, list) else []


class ChunkCache:
    """
    LRU cache for CSV chunks with RAM limit.

    Keeps recently accessed chunks in memory to avoid repeated disk reads.
    Automatically evicts old chunks when RAM limit is reached.
    """

    def __init__(self, max_ram_mb: int = 4096):
        """
        Initialize chunk cache.

        Args:
            max_ram_mb: Maximum RAM to use in MB (default: 4GB)
        """
        self.max_ram_bytes = max_ram_mb * 1024 * 1024
        self.cache = OrderedDict()  # LRU cache: (file_path, chunk_idx) -> DataFrame
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def get(self, file_path: str, chunk_idx: int) -> Optional[pd.DataFrame]:
        """
        Get chunk from cache.

        Returns None if not in cache (cache miss).
        """
        key = (file_path, chunk_idx)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, file_path: str, chunk_idx: int, chunk: pd.DataFrame):
        """
        Add chunk to cache, evicting old chunks if needed.
        """
        key = (file_path, chunk_idx)

        # Estimate chunk size in bytes
        chunk_size = chunk.memory_usage(deep=True).sum()

        # Evict old chunks if needed
        while self.current_size + chunk_size > self.max_ram_bytes and len(self.cache) > 0:
            # Remove least recently used (first item)
            old_key, old_chunk = self.cache.popitem(last=False)
            old_size = old_chunk.memory_usage(deep=True).sum()
            self.current_size -= old_size

        # Add new chunk
        if chunk_size <= self.max_ram_bytes:  # Only cache if it fits
            self.cache[key] = chunk
            self.current_size += chunk_size

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.current_size = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'num_chunks': len(self.cache),
            'ram_usage_mb': self.current_size / (1024 * 1024)
        }


def dfs_subgraph(graph: Dict[int, List[int]], start_node: int, depth: int = 1) -> set:
    """
    Extract subgraph using DFS up to specified depth.

    Args:
        graph: Adjacency list representation
        start_node: Starting node
        depth: Maximum depth to explore

    Returns:
        Set of nodes in the subgraph

    Note:
        This function is used in conjunction with dual extraction (from both node_a and node_b)
        and union operation to ensure sufficient context for both cyclic and linear structures.

        Examples with depth=1:
        - Triangle (C-C-O): When predicting C-C bond, union of neighborhoods captures all 3 nodes
        - Linear chain (O-C-C-O): When predicting C-C bond, union captures 2 hops (4 nodes total)
    """
    visited = set()
    queue = deque([(start_node, 0)])  # (node, current_depth)
    subgraph_nodes = set()

    while queue:
        node, curr_depth = queue.popleft()
        if node in visited or curr_depth > depth:
            continue

        visited.add(node)
        subgraph_nodes.add(node)

        if curr_depth < depth:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, curr_depth + 1))

    return subgraph_nodes


class MultiFileStreamingDataset(Dataset):
    """
    Streaming dataset for TB-scale molecular geometry data.

    Key Features:
    - NO RAM caching - streams everything from disk
    - File-based epochs: each file = 1 epoch
    - Chunked CSV reading for memory efficiency
    - Stereochemistry-aware geometry computation
    - Supports multiple file patterns (e.g., "Compound_*.csv")

    Usage:
        >>> dataset = MultiFileStreamingDataset(
        ...     file_pattern="../Compound_*.csv",
        ...     element_to_idx=element_mapping,
        ...     current_file_idx=0
        ... )
        >>> dataset.next_file()  # Cycle to next file
        >>> data = dataset[0]    # Stream one sample
    """

    def __init__(
        self,
        file_paths: List[str],
        element_to_idx: Dict[str, int],
        current_file_idx: int = 0,
        max_bonds: int = 7,
        bond_threshold: float = 4.0,
        subgraph_depth: int = 1,
        chunk_size: int = 10000,
        include_non_bonded: bool = False,
        training: bool = True,
        max_samples: Optional[int] = None,
        cache_ram_mb: int = 4096,
        split: Optional[str] = None,
        split_seed: int = 42,
        kfold: int = 5,
        current_fold: int = 0
    ):
        """
        Initialize multi-file streaming dataset.

        Args:
            file_paths: List of CSV file paths
            element_to_idx: Mapping from element symbols to indices
            current_file_idx: Index of current file to use (for file-based epochs)
            max_bonds: Maximum number of bonds per atom to predict
            bond_threshold: Distance threshold for non-bonded pairs (Angstroms)
            subgraph_depth: Depth for local subgraph extraction
            chunk_size: Number of rows to read at once from CSV
            include_non_bonded: Include non-bonded pairs for distance diversity
            training: If True, apply random rotation augmentation
            max_samples: Maximum number of samples (bonds) to use (for testing)
            cache_ram_mb: Maximum RAM for caching chunks (MB, default: 4GB)
            split: Which split to use - 'train', 'val', or None (use all data)
            split_seed: Random seed for k-fold splitting (for reproducibility)
            kfold: Number of folds for k-fold cross-validation (default: 5)
            current_fold: Which fold to use for validation (0 to kfold-1)
        """
        super().__init__()

        # Initialize chunk cache
        self.chunk_cache = ChunkCache(max_ram_mb=cache_ram_mb)

        # File management
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        if not self.file_paths:
            raise ValueError(f"No files provided")

        self.current_file_idx = current_file_idx % len(self.file_paths)
        self.current_file = self.file_paths[self.current_file_idx]

        # Dataset parameters
        self.element_to_idx = element_to_idx
        self.max_bonds = max_bonds
        self.bond_threshold = bond_threshold
        self.subgraph_depth = subgraph_depth
        self.chunk_size = chunk_size
        self.include_non_bonded = include_non_bonded
        self.training = training
        self.max_samples = max_samples
        self.split = split
        self.split_seed = split_seed
        self.kfold = kfold
        self.current_fold = current_fold

        # Index mapping (lightweight - only counts, not data)
        self.valid_molecule_indices = []  # (chunk_idx, row_in_chunk, bond_count)
        self.total_bonds = 0

        # Load current file metadata
        self._load_file_metadata()

        print(f"Dataset initialized with {len(self.file_paths)} files")
        print(f"Current file: {self.current_file}")
        print(f"Found {len(self.valid_molecule_indices)} valid molecules with {self.total_bonds} bonds")

    def _load_file_metadata(self):
        """
        Load metadata for current file (molecule counts, bond counts).

        Only uses PubChem columns.
        Skips any row that does not have valid PubChem mapping/data.
        """
        print(f"\nLoading metadata from: {self.current_file}")

        all_molecule_indices = []
        chunk_idx = 0

        # Read CSV in chunks and collect all molecule indices
        for chunk in pd.read_csv(self.current_file, chunksize=self.chunk_size):
            for row_in_chunk, (_, row) in enumerate(chunk.iterrows()):
                try:
                    # Always need GEOMETRY_COLUMN for 3D coordinates
                    if GEOMETRY_COLUMN not in row or pd.isna(row[GEOMETRY_COLUMN]):
                        continue

                    # Require valid PubChem mapping
                    if (
                        'Pubchem_to_xyz_aid_mapping' not in row
                        or pd.isna(row['Pubchem_to_xyz_aid_mapping'])
                        or _pubchem_mapping_has_error(row['Pubchem_to_xyz_aid_mapping'])
                    ):
                        continue

                    # Parse 3D coordinates from XYZ
                    coordinates, xyz_elements = parse_xyz_string(row[GEOMETRY_COLUMN])
                    if not coordinates or len(coordinates) < 2:
                        continue

                    # Use PubChem atoms and bonds only
                    coordinates, elements, bonds_data = parse_pubchem_data(row, coordinates)
                    if not coordinates or not bonds_data:
                        continue

                    # Count valid bonds
                    bond_count = 0
                    for bond in bonds_data:
                        if 'i' in bond and 'j' in bond:
                            i, j = bond['i'], bond['j']
                            if i < len(coordinates) and j < len(coordinates) and i != j:
                                bond_count += 1

                    # Add non-bonded pairs count if requested
                    if self.include_non_bonded and bond_count > 0:
                        bond_count += bond_count // 3

                    if bond_count > 0:
                        all_molecule_indices.append((chunk_idx, row_in_chunk, bond_count))

                except Exception:
                    continue

            chunk_idx += 1

        # Apply k-fold train/val split
        if self.split is not None:
            self.valid_molecule_indices = self._apply_kfold_split(all_molecule_indices)
        else:
            self.valid_molecule_indices = all_molecule_indices

        # Calculate total bonds
        self.total_bonds = sum(bond_count for _, _, bond_count in self.valid_molecule_indices)

        # Apply max_samples limit if specified
        if self.max_samples and self.total_bonds > self.max_samples:
            total = 0
            for idx, (chunk_idx, row_in_chunk, bond_count) in enumerate(self.valid_molecule_indices):
                total += bond_count
                if total >= self.max_samples:
                    self.valid_molecule_indices = self.valid_molecule_indices[:idx+1]
                    self.total_bonds = self.max_samples
                    break

        split_info = f" ({self.split} split)" if self.split else ""
        print(f"Metadata loaded{split_info}: {len(self.valid_molecule_indices)} molecules, {self.total_bonds} bonds")

    def _apply_kfold_split(self, all_molecule_indices: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Apply k-fold cross-validation split to molecule indices.

        Special case: if current_fold == -1, train on all data (val is empty).

        Args:
            all_molecule_indices: List of (chunk_idx, row_in_chunk, bond_count) tuples

        Returns:
            Filtered list of molecule indices for the specified fold
        """
        if not all_molecule_indices:
            return []

        # Special case: current_fold == -1 means train on all data, no validation
        if self.current_fold == -1:
            if self.split == 'train':
                return all_molecule_indices
            elif self.split == 'val':
                return []  # Empty validation set
            else:
                return all_molecule_indices

        # Use a deterministic random seed for reproducible splits
        rng = random.Random(self.split_seed + hash(self.current_file))

        # Shuffle indices with the deterministic seed
        shuffled_indices = list(all_molecule_indices)
        rng.shuffle(shuffled_indices)

        n_total = len(shuffled_indices)
        fold_size = n_total // self.kfold

        # Calculate fold boundaries
        val_start = self.current_fold * fold_size
        if self.current_fold == self.kfold - 1:
            # Last fold gets remaining samples
            val_end = n_total
        else:
            val_end = val_start + fold_size

        # Return appropriate split (no overlap between train and val)
        if self.split == 'train':
            # All indices except the validation fold
            train_indices = shuffled_indices[:val_start] + shuffled_indices[val_end:]
            return train_indices
        elif self.split == 'val':
            # Only the validation fold
            return shuffled_indices[val_start:val_end]
        else:
            return all_molecule_indices

    def next_file(self) -> str:
        """
        Cycle to the next file (for file-based epochs).

        Returns:
            Path to new current file
        """
        # Print cache stats before switching
        cache_stats = self.chunk_cache.get_stats()
        print(f"\nCache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
              f"hit rate: {cache_stats['hit_rate']:.2%}, RAM: {cache_stats['ram_usage_mb']:.1f} MB")

        # Clear cache when switching files to free memory
        self.chunk_cache.clear()

        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_paths)
        self.current_file = self.file_paths[self.current_file_idx]

        print(f"\nSwitching to file {self.current_file_idx + 1}/{len(self.file_paths)}: {self.current_file}")

        # Reload metadata for new file
        self._load_file_metadata()

        return self.current_file

    def get_current_file_info(self) -> Dict:
        """Get information about current file."""
        return {
            'file_idx': self.current_file_idx,
            'file_path': self.current_file,
            'total_files': len(self.file_paths),
            'num_molecules': len(self.valid_molecule_indices),
            'num_bonds': self.total_bonds
        }

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.chunk_cache.get_stats()
    def __len__(self) -> int:
        """Return total number of training samples (bonds) in current file."""
        if self.max_samples:
            return min(self.total_bonds, self.max_samples)
        return self.total_bonds

    def _get_molecule_and_bond_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Convert flat bond index to (chunk_idx, row_in_chunk, bond_idx).

        Args:
            idx: Flat index into all bonds in current file

        Returns:
            chunk_idx: Which chunk this molecule is in
            row_in_chunk: Row index within that chunk
            bond_idx: Which bond within this molecule
        """
        current_sum = 0
        for chunk_idx, row_in_chunk, bond_count in self.valid_molecule_indices:
            if idx < current_sum + bond_count:
                bond_idx = idx - current_sum
                return chunk_idx, row_in_chunk, bond_idx
            current_sum += bond_count

        raise IndexError(f"Index {idx} out of range (total: {self.total_bonds})")

    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample, using RAM cache to avoid repeated disk reads.

        Args:
            idx: Index of bond to load

        Returns:
            PyTorch Geometric Data object
        """
        # Find which molecule and bond
        chunk_idx, row_in_chunk, bond_idx = self._get_molecule_and_bond_index(idx)

        # Try to get chunk from cache
        chunk_to_read = self.chunk_cache.get(self.current_file, chunk_idx)

        if chunk_to_read is None:
            # Cache miss - read from disk
            skip_chunks = chunk_idx
            chunk_to_read = pd.read_csv(
                self.current_file,
                skiprows=range(1, skip_chunks * self.chunk_size + 1),  # Skip header + previous chunks
                nrows=self.chunk_size
            )
            # Add to cache
            self.chunk_cache.put(self.current_file, chunk_idx, chunk_to_read)

        # Get the specific row
        row = chunk_to_read.iloc[row_in_chunk]

        # Process this molecule and bond
        data = self._process_molecule_bond(row, bond_idx)

        return data

    def _process_molecule_bond(self, row, bond_idx: int) -> Data:
        """
        Process a specific molecule and return the bond at bond_idx.

        This is where stereochemistry-aware geometry computation happens.

        Args:
            row: DataFrame row containing molecule data
            bond_idx: Index of bond to process

        Returns:
            PyTorch Geometric Data object with local geometry

        Subgraph Extraction Strategy:
            The function extracts local subgraphs around both target atoms (A and B) and takes
            their union to ensure sufficient context for geometry prediction.

            With depth=1 (default), this approach handles different topologies correctly:

            Case 1 - Cyclic structures (e.g., triangle C-C-O):
                    C(0)
                   /    \\
                  C(1)--O(2)

                When predicting C(0)-C(1) bond:
                - subgraph_a from C(0): {C(0), C(1), O(2)}
                - subgraph_b from C(1): {C(1), C(0), O(2)}
                - Union: {C(0), C(1), O(2)} - captures entire triangle
                - All edges preserved with bond types: C-C, C-O, O-C

            Case 2 - Linear chains (e.g., O-C-C-O):
                O(0)--C(1)--C(2)--O(3)

                When predicting C(1)-C(2) bond:
                - subgraph_a from C(1): {C(1), O(0), C(2)}
                - subgraph_b from C(2): {C(2), C(1), O(3)}
                - Union: {O(0), C(1), C(2), O(3)} - captures 2-hop neighborhood
                - All edges preserved with bond types: O-C, C-C, C-O

            The dual extraction + union strategy ensures that:
            1. Cyclic structures: All atoms in local cycles are captured
            2. Linear structures: Context from both sides of the bond is included
            3. Bond types: Preserved via edge_attr for all edges in subgraph
            4. Scalability: Memory-efficient for large molecules (only local context extracted)
        """
        # Parse 3D coordinates from XYZ (always needed for geometry)
        xyz_coordinates, xyz_elements = parse_xyz_string(row[GEOMETRY_COLUMN])

        # Determine whether to use PubChem columns
        use_pubchem = (
            'Pubchem_to_xyz_aid_mapping' in row
            and not _pubchem_mapping_has_error(row.get('Pubchem_to_xyz_aid_mapping'))
        )

        if use_pubchem:
            coordinates, elements, bonds_data = parse_pubchem_data(row, xyz_coordinates)
            # If PubChem parsing failed for any reason, fall back
            if not coordinates or not bonds_data:
                coordinates, elements = xyz_coordinates, xyz_elements
                bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])
        else:
            coordinates, elements = xyz_coordinates, xyz_elements
            bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])

        # Filter valid elements
        valid_indices = [i for i, elem in enumerate(elements) if elem.strip()]
        coordinates = [coordinates[i] for i in valid_indices]
        elements = [elements[i] for i in valid_indices]

        # Create element indices using the provided mapping
        element_indices = []
        for elem in elements:
            elem_idx = self.element_to_idx.get(elem, 0)  # 0 for 'OTHER'
            element_indices.append(elem_idx)
        x = torch.tensor(element_indices, dtype=torch.long)

        pos = torch.tensor(coordinates, dtype=torch.float)

        # Create adjacency graph from dataset bonds
        # Also track bond types: 0=single, 1=aromatic, 2=double, 3=triple
        adjacency = defaultdict(list)
        bond_pairs = []
        bond_types_dict = {}  # (i, j) -> bond_type

        # Add bonds from dataset
        for bond in bonds_data:
            if 'i' in bond and 'j' in bond:
                i, j = bond['i'], bond['j']
                if i < len(coordinates) and j < len(coordinates) and i != j:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    dist = torch.norm(pos[i] - pos[j])
                    bond_pairs.append((i, j, dist.item()))

                    # Extract bond type from order
                    bond_order = bond.get('order', 1.0)
                    if bond_order == 1.0:
                        bond_type = 0  # Single bond
                    elif bond_order == 1.5:
                        bond_type = 1  # Aromatic bond
                    elif bond_order == 2.0:
                        bond_type = 2  # Double bond
                    elif bond_order == 3.0:
                        bond_type = 3  # Triple bond
                    else:
                        bond_type = 0  # Default to single for unknown orders

                    # Store bond type for both directions
                    bond_types_dict[(i, j)] = bond_type
                    bond_types_dict[(j, i)] = bond_type

        # Add non-bonded pairs if requested
        if self.include_non_bonded:
            bonded_set = set((min(i,j), max(i,j)) for i,j,_ in bond_pairs)
            non_bonded_pairs = []

            for i in range(len(coordinates)):
                for j in range(i+1, len(coordinates)):
                    if (i,j) not in bonded_set:
                        dist = torch.norm(pos[i] - pos[j])
                        if dist < self.bond_threshold:
                            non_bonded_pairs.append((i, j, dist.item()))

            if non_bonded_pairs:
                n_sample = min(len(bond_pairs) // 3, len(non_bonded_pairs))
                sampled_non_bonded = random.sample(non_bonded_pairs, n_sample)
                bond_pairs.extend(sampled_non_bonded)

        # Get the specific bond
        if bond_idx >= len(bond_pairs):
            raise IndexError(f"Bond index {bond_idx} out of range")

        node_a, node_b, target_distance = bond_pairs[bond_idx]

        # Extract subgraph around both nodes
        subgraph_a = dfs_subgraph(adjacency, node_a, depth=self.subgraph_depth)
        subgraph_b = dfs_subgraph(adjacency, node_b, depth=self.subgraph_depth)
        subgraph_nodes = subgraph_a.union(subgraph_b)
        subgraph_nodes.add(node_a)
        subgraph_nodes.add(node_b)

        # Create node mapping
        node_list = sorted(list(subgraph_nodes))
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_list)}

        # Map target nodes
        new_node_a = node_mapping[node_a]
        new_node_b = node_mapping[node_b]

        # Create subgraph features and positions
        subgraph_x = x[node_list]
        subgraph_pos = pos[node_list]

        # Create subgraph edges with bond type attributes
        edge_index = []
        edge_attr = []  # Bond type for each edge
        for old_i in node_list:
            for old_j in adjacency[old_i]:
                if old_j in node_mapping:
                    new_i = node_mapping[old_i]
                    new_j = node_mapping[old_j]
                    edge_index.append([new_i, new_j])

                    # Get bond type (default to 0 if not found - for non-bonded pairs)
                    bond_type = bond_types_dict.get((old_i, old_j), 0)
                    edge_attr.append(bond_type)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long) if edge_attr else torch.empty((0,), dtype=torch.long)

        # Create subgraph elements list (for deterministic neighbor ordering)
        subgraph_elements = [elements[old_idx] for old_idx in node_list]

        # Create bond orders dict for subgraph (new indices)
        # bond_types_dict has 0=single, 1=aromatic, 2=double, 3=triple
        # Convert to bond order: 0->1.0, 1->1.5, 2->2.0, 3->3.0
        bond_type_to_order = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
        subgraph_bond_orders = {}
        for (old_i, old_j), btype in bond_types_dict.items():
            if old_i in node_mapping and old_j in node_mapping:
                new_i, new_j = node_mapping[old_i], node_mapping[old_j]
                subgraph_bond_orders[(new_i, new_j)] = bond_type_to_order.get(btype, 1.0)

        # Compute chirality sign for every subgraph node using full-molecule context.
        # Using all direct neighbours (not just those in the subgraph) keeps the
        # signed volume consistent regardless of subgraph depth.
        chirality_values = []
        for node_global in node_list:
            global_neighbors = adjacency[node_global]
            if len(global_neighbors) < 3:
                chirality_values.append(0.0)
                continue
            global_bond_orders = {
                (node_global, n): bond_type_to_order.get(bond_types_dict.get((node_global, n), 0), 1.0)
                for n in global_neighbors
            }
            sort_keys = get_neighbor_sort_keys(global_neighbors, elements, global_bond_orders, node_global)
            sorted_nbrs = [n for n, _ in sorted(zip(global_neighbors, sort_keys), key=lambda kv: kv[1])]
            chirality_values.append(compute_chirality_sign(pos, node_global, sorted_nbrs))
        chirality = torch.tensor(chirality_values, dtype=torch.float)

        # **NEW: Compute stereochemistry-aware local geometry**
        # For node A, use A→B bond as reference
        node_a_neighbors = [node_mapping[n] for n in adjacency[node_a] if n in node_mapping]

        if new_node_b in node_a_neighbors:
            # Create deterministic sort keys for neighbors of A
            sort_keys_a = get_neighbor_sort_keys(
                node_a_neighbors, subgraph_elements, subgraph_bond_orders, new_node_a
            )
            # Compute local geometry for node A
            ordered_neighbors_a, geometry_a, frame_a = compute_local_geometry(
                subgraph_pos,
                center_idx=new_node_a,
                neighbor_indices=node_a_neighbors,
                reference_idx=new_node_b,
                augment_rotation=self.training,  # Random rotation during training
                neighbor_sort_keys=sort_keys_a  # Deterministic ordering
            )
        else:
            # Fallback: no local geometry
            ordered_neighbors_a = []
            geometry_a = []
            frame_a = None

        # For node B, use B→A bond as reference
        node_b_neighbors = [node_mapping[n] for n in adjacency[node_b] if n in node_mapping]

        if new_node_a in node_b_neighbors:
            # Create deterministic sort keys for neighbors of B
            sort_keys_b = get_neighbor_sort_keys(
                node_b_neighbors, subgraph_elements, subgraph_bond_orders, new_node_b
            )
            # Compute local geometry for node B
            ordered_neighbors_b, geometry_b, frame_b = compute_local_geometry(
                subgraph_pos,
                center_idx=new_node_b,
                neighbor_indices=node_b_neighbors,
                reference_idx=new_node_a,
                augment_rotation=self.training,
                neighbor_sort_keys=sort_keys_b  # Deterministic ordering
            )
        else:
            # Fallback: no local geometry
            ordered_neighbors_b = []
            geometry_b = []
            frame_b = None

        # Create bond masks first (needed for relative azimuthal conversion)
        bond_mask_a = np.zeros(self.max_bonds)
        for i in range(min(len(geometry_a), self.max_bonds)):
            bond_mask_a[i] = 1.0

        bond_mask_b = np.zeros(self.max_bonds)
        for i in range(min(len(geometry_b), self.max_bonds)):
            bond_mask_b[i] = 1.0

        # Pack geometry into tensors (θ, Δφ) for up to max_bonds
        # NOTE: convert angles from radians to degrees for model/loss compatibility
        # NOTE: Convert absolute φ to relative Δφ (first neighbor Δφ=0)

        # Process geometry_a: convert to (r, θ°, φ°) first
        geometry_a_deg = []
        for i in range(self.max_bonds):
            if i < len(geometry_a):
                r, theta, phi = geometry_a[i]
                theta_deg = math.degrees(theta)
                phi_deg = math.degrees(phi)
                geometry_a_deg.append([r, theta_deg, phi_deg])
            else:
                geometry_a_deg.append([0.0, 0.0, 0.0])  # Padding

        # Convert to numpy for processing
        geometry_a_np = np.array(geometry_a_deg)
        # Convert (r, θ, φ) to (r, θ, Δφ) using relative azimuthal
        geometry_a_relative = convert_to_relative_azimuthal(geometry_a_np, bond_mask_a)
        # Extract only (θ, Δφ) - discard r
        geometry_a_padded = geometry_a_relative[:, 1:].tolist()  # Shape: (max_bonds, 2)

        # Process geometry_b similarly
        geometry_b_deg = []
        for i in range(self.max_bonds):
            if i < len(geometry_b):
                r, theta, phi = geometry_b[i]
                theta_deg = math.degrees(theta)
                phi_deg = math.degrees(phi)
                geometry_b_deg.append([r, theta_deg, phi_deg])
            else:
                geometry_b_deg.append([0.0, 0.0, 0.0])  # Padding

        # Convert to numpy for processing
        geometry_b_np = np.array(geometry_b_deg)
        # Convert (r, θ, φ) to (r, θ, Δφ) using relative azimuthal
        geometry_b_relative = convert_to_relative_azimuthal(geometry_b_np, bond_mask_b)
        # Extract only (θ, Δφ) - discard r
        geometry_b_padded = geometry_b_relative[:, 1:].tolist()  # Shape: (max_bonds, 2)

        # Convert masks to tensors
        bond_mask_a = torch.tensor(bond_mask_a, dtype=torch.float)
        bond_mask_b = torch.tensor(bond_mask_b, dtype=torch.float)

        # Create data object
        data = Data(
            x=subgraph_x,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Bond type: 0=single, 1=aromatic, 2=double, 3=triple
            pos=subgraph_pos,
            chirality=chirality,  # (N_subgraph,) float: +1 R, -1 S, 0 achiral/planar
            node_a=new_node_a,
            node_b=new_node_b,
            target_distance=torch.tensor([target_distance], dtype=torch.float),
            target_geometry_a=torch.tensor(geometry_a_padded, dtype=torch.float),  # (max_bonds, 2): (θ°, Δφ°)
            target_geometry_b=torch.tensor(geometry_b_padded, dtype=torch.float),  # (max_bonds, 2): (θ°, Δφ°)
            bond_mask_a=bond_mask_a,
            bond_mask_b=bond_mask_b
        )

        return data


def load_elements_from_txt(txt_path: str) -> Dict[str, int]:
    """
    Load element vocabulary from a text file.

    Each line in the file should contain one element symbol.
    Index 0 is implicitly reserved for 'OTHER' (unknown elements).
    Elements from file start at index 1.

    Args:
        txt_path: Path to text file containing element symbols (one per line)

    Returns:
        element_to_idx: Mapping from element to index (0 = 'OTHER', 1+ = known elements)

    Example txt file:
        H
        C
        N
        O
    """
    # Start with OTHER at index 0 (implicit)
    element_to_idx = {'OTHER': 0}

    with open(txt_path, 'r') as f:
        for line in f:
            element = line.strip()
            if element and element != 'OTHER':  # Skip empty lines and explicit OTHER
                # Add element starting from index 1
                if element not in element_to_idx:
                    element_to_idx[element] = len(element_to_idx)

    print(f"Loaded {len(element_to_idx)} elements from {txt_path} (including implicit OTHER)")
    print(f"  Known elements: {sorted([e for e in element_to_idx.keys() if e != 'OTHER'])}")

    return element_to_idx


def create_elements_txt_from_data(
    file_paths: List[str],
    output_path: str = "elements.txt",
    max_molecules: int = 100000
) -> Dict[str, int]:
    """
    Scan dataset files to extract unique elements and create elements.txt file.

    Reads files in chunks and stops after max_molecules for efficiency.

    Args:
        file_paths: List of CSV file paths to scan
        output_path: Path to save elements.txt (default: "elements.txt")
        max_molecules: Maximum molecules to scan (for speed)

    Returns:
        element_to_idx: Mapping from element to index (0 = 'OTHER')
    """
    unique_elements = set()
    files = file_paths if isinstance(file_paths, list) else [file_paths]

    if not files:
        raise ValueError(f"No files provided")

    print(f"Scanning {len(files)} file(s) to extract unique elements...")

    molecules_scanned = 0

    for file_path in files:
        print(f"  Reading: {file_path}")

        for chunk in pd.read_csv(file_path, chunksize=10000):
            for _, row in chunk.iterrows():
                try:
                    if GEOMETRY_COLUMN not in row or pd.isna(row[GEOMETRY_COLUMN]):
                        continue

                    coordinates, elements = parse_xyz_string(row[GEOMETRY_COLUMN])
                    if not elements:
                        continue

                    for element in elements:
                        unique_elements.add(element.strip())

                    molecules_scanned += 1

                    if molecules_scanned >= max_molecules:
                        break

                except Exception:
                    continue

            if molecules_scanned >= max_molecules:
                break

        if molecules_scanned >= max_molecules:
            break

    # Sort elements alphabetically for consistency
    sorted_elements = sorted(unique_elements)

    # Write to file (no OTHER - it's implicit at index 0)
    with open(output_path, 'w') as f:
        for element in sorted_elements:
            f.write(f"{element}\n")

    print(f"\nCreated {output_path} with {len(sorted_elements)} elements (OTHER is implicit at index 0)")
    print(f"  Scanned {molecules_scanned} molecules")
    print(f"  Unique elements found: {sorted_elements}")

    # Return the mapping
    return load_elements_from_txt(output_path)


def get_element_vocab(
    txt_path: str = "elements.txt",
    file_paths: Optional[List[str]] = None,
    max_molecules: int = 100000
) -> Dict[str, int]:
    """
    Get element vocabulary, loading from txt file or creating it if needed.

    Args:
        txt_path: Path to elements.txt file
        file_paths: List of CSV files to scan if txt doesn't exist
        max_molecules: Max molecules to scan when creating txt

    Returns:
        element_to_idx: Mapping from element to index (0 = 'OTHER')

    Usage:
        # Try to load, create if missing
        element_vocab = get_element_vocab(
            txt_path="elements.txt",
            file_paths=["data/file1.csv", "data/file2.csv"]
        )

        # Load only (error if missing)
        element_vocab = get_element_vocab(txt_path="elements.txt")
    """
    if os.path.exists(txt_path):
        print(f"Loading element vocabulary from {txt_path}")
        return load_elements_from_txt(txt_path)
    else:
        if file_paths is None:
            raise FileNotFoundError(
                f"Element vocabulary file not found: {txt_path}\n"
                f"Please provide file_paths to scan and create it."
            )

        print(f"Element vocabulary file not found: {txt_path}")
        print(f"Creating it by scanning data files...")
        return create_elements_txt_from_data(file_paths, txt_path, max_molecules)
